import abc
from typing import Sequence

import torch
from torch import nn, optim


# ---------------
# Base class
# ---------------
class MultiTaskStrategy(abc.ABC):
    """
    Base class for multi-task learning (MTL)

    This class is meant to 'hide' a lot of code for computing gradients and apply them with an optimiser
    """

    __slots__ = ("_optimiser",)

    def __init__(self, optimiser: optim.Optimizer):
        self._optimiser = optimiser

    # --------------
    # Common steps
    # --------------
    @abc.abstractmethod
    def backward(self, *, losses: Sequence[torch.Tensor], model: torch.nn.Module):
        """Method for computing gradients. Replaces loss.backwards()"""

    def step(self):
        self._optimiser.step()

    def zero_grad(self):
        self._optimiser.zero_grad()


# ---------------
# Implementations
# ---------------
class PCGrad(MultiTaskStrategy):
    """
    Implementation of PCGrad

    Paper:
        Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient surgery for multi-task
        learning. Advances in neural information processing systems, 33, 5824-5836.
    """

    def backward(self, *, losses: Sequence[torch.Tensor], model: torch.nn.Module):
        """Following 'Algorithm 1: PCGrad Update Rule' in the paper"""
        num_tasks = len(losses)
        trainable_params = [params for params in model.parameters() if params.requires_grad]

        # --------------
        # Compute gradients for all tasks (steps 1-2)
        # --------------
        gradients = []
        gradients_pc = []
        for loss in losses:
            # Compute gradients
            self.zero_grad()
            loss.backward(retain_graph=True)

            # Flatten gradients
            grads = [params.grad.view(-1) if params.grad is not None else torch.zeros_like(params).view(-1)
                     for params in trainable_params]

            grad = torch.cat(grads)
            gradients.append(grad)
            gradients_pc.append(grad.clone())

        # --------------
        # Modify gradients if conflicting (steps 3-7)
        # --------------
        for task_i in range(num_tasks):
            other_tasks = [task for task in range(num_tasks) if task != task_i]  # Sub-optimal implementation
            for task_j in other_tasks:
                # Compute dot product
                grad_dot_product = torch.dot(gradients_pc[task_i], gradients[task_j])

                if grad_dot_product < 0:
                    # Subtract the projection
                    factor = grad_dot_product / (torch.norm(gradients[task_j]) ** 2 + 1e-10)
                    gradients_pc[task_i] -= factor * gradients[task_j]

        # --------------
        # Update gradients (step 8)
        # --------------
        # Sum the gradients
        final_gradients = torch.stack(gradients_pc).sum(dim=0)

        # Set the gradients
        offset = 0
        for params in trainable_params:
            num_elements = params.numel()
            params.grad = final_gradients[offset:(offset + num_elements)].view_as(params).clone()
            offset += num_elements


class GradNorm(MultiTaskStrategy):
    """
    Implementation of GradNorm

    Note that the model used with this class must implement .gradnorm_parameters(), which provides the network weights
    where GradNorm should be applied (usually the final shared layer, according to Algorithm 1 in the paper)

    Paper:
        Chen, Z., Badrinarayanan, V., Lee, C. Y., & Rabinovich, A. (2018, July). Gradnorm: Gradient normalization for
        adaptive loss balancing in deep multitask networks. In International conference on machine learning
        (pp. 794-803). PMLR.
    """

    __slots__ = ("_alpha", "_initial_losses", "_loss_weights", "_learning_rate")

    def __init__(self, optimiser, *, alpha, learning_rate):
        """
        Initialise

        Parameters
        ----------
        optimiser : optim.Optimizer
        alpha : float
            The alpha HP of the algorithm. The values 1.5 and 1.2 were good in the original paper, but any values
            0 < alpha < 3 outperformed equal weights baseline (see Sec. 5.4 in the paper)
        learning_rate : float
            The loss weights are updated with SGD, using this learning rate. ChatGPT suggested 0.025
        """
        super().__init__(optimiser)

        self._alpha = alpha
        self._learning_rate = learning_rate
        self._initial_losses = None
        self._loss_weights = None  # Will be initialised as ones in the first pass

    def to(self, device):
        if self._loss_weights is not None:
            self._loss_weights = self._loss_weights.to(device)

    def backward(self, losses: Sequence[torch.Tensor], model: nn.Module):
        """Following 'Algorithm 1 Training with GradNorm' in the paper."""
        device = losses[0].device
        num_tasks = len(losses)

        # Maybe initialise loss weights
        if self._loss_weights is None:
            self._loss_weights = torch.nn.Parameter(torch.ones(num_tasks, device=device), requires_grad=True)

        # Compute weighted losses as a generator
        weighted_losses = (weight * loss for weight, loss in zip(self._loss_weights, losses))

        # --------------
        # Compute gradient norms and everything required for updating loss weights
        # --------------
        # Gradient norms for all tasks
        gradient_norms = []  # G_W^{(i)} in the algorithm
        for loss in weighted_losses:
            self.zero_grad()
            loss.backward(retain_graph=True)
            grads = torch.autograd.grad(outputs=loss,inputs=model.gradnorm_parameters(), retain_graph=True,
                                        create_graph=True)  # Needed to allow backprop through gradnorm_loss
            grads = torch.cat([grad.view(-1) for grad in grads])
            gradient_norms.append(torch.norm(grads))
        gradient_norms = torch.stack(gradient_norms)

        # Average gradient norm
        avg_grad_norm = gradient_norms.mean()  # \overline{G}_W in the algorithm

        # Capture initial losses
        if self._initial_losses is None:
            with torch.no_grad():
                self._initial_losses = torch.tensor([loss.item() for loss in losses], device=device)  # L_i(0)

        # Loss ratios. \tilde{L}_i(t) in the algorithm
        loss_ratios = torch.tensor([loss.item() for loss in losses], device=device) / self._initial_losses

        # Relative inverse training rates
        relative_inv_train_rate = loss_ratios / loss_ratios.mean()  # r_i(t) in the algorithm

        # Compute GradNorm loss. Detach because only gradients of the loss weights should be computed
        target_norms = avg_grad_norm * (relative_inv_train_rate ** self._alpha)
        gradnorm_loss = nn.L1Loss()(gradient_norms, target_norms.detach())  # L_{grad}

        # Compute GradNorm gradients
        self.zero_grad()
        gradnorm_loss.backward(retain_graph=True)

        # --------------
        # Gradients for the model parameters and loss weights
        # --------------
        # Model parameters
        self.zero_grad()
        total_model_loss = torch.stack(
            [weights * loss for weights, loss in zip(self._loss_weights.detach(), losses)]).sum()
        total_model_loss.backward()

        # Loss weights. Must do this manually as the parameters are not part of the optimiser
        with (torch.no_grad()):
            self._loss_weights.data -= self._learning_rate * self._loss_weights.grad
            self._loss_weights.grad.zero_()
            self._loss_weights.data.clamp(min=1e-6)  # Weights should be positive

            # Re-normalisation
            self._loss_weights.data *= num_tasks / self._loss_weights.data.sum()


class UncertaintyWeighting(MultiTaskStrategy):
    ...
