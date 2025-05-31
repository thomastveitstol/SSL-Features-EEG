import abc
from typing import Sequence

import torch


# ---------------
# Base class
# ---------------
class MultiTaskStrategy(abc.ABC):
    """
    Base class for multi-task learning (MTL)

    This class is meant to 'hide' a lot of code for computing gradients and apply them with an optimiser
    """

    __slots__ = ("_optimiser",)

    def __init__(self, optimiser: torch.optim.Optimizer):
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
    ...


class UncertaintyWeighting(MultiTaskStrategy):
    ...
