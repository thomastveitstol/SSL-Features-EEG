import abc
from typing import Sequence, Optional, Union, Tuple, Type, Any, Dict

import torch
from torch import nn, optim

from elecssl.models.mtl_strategies.minimum_norm_solver import MinNormSolver


# ---------------
# Base class
# ---------------
class MultiTaskStrategy(abc.ABC):
    """
    Base class for multi-task learning (MTL)

    This class is meant to 'hide' a lot of code for computing gradients and apply them with an optimiser
    """

    __slots__ = ("_optimiser", "_model")

    def __init__(self, optimiser: optim.Optimizer, model: nn.Module, **kwargs):
        self._optimiser = optimiser
        self._model = model

    # --------------
    # Common steps
    # --------------
    @abc.abstractmethod
    def backward(self, *, losses: Sequence[torch.Tensor]):
        """Method for computing gradients. Replaces loss.backwards()"""

    def step(self):
        self._optimiser.step()

    def zero_grad(self):
        self._optimiser.zero_grad()

    # --------------
    # For hyperparameter optimisation
    # --------------
    @staticmethod
    @abc.abstractmethod
    def suggest_hyperparameters(name, trial, config) -> Dict[str, Any]:
        """Method for sampling hyperparameter configurations, given distributions"""


# ---------------
# Implementations
# ---------------
class EqualWeighting(MultiTaskStrategy):
    """
    Implementation of equal weighting for multi-task learning. This can also be used for single-task learning.

    Each task contributes equally to the total loss.
    """
    __slots__ = ()

    def backward(self, *, losses: Union[Sequence[torch.Tensor], torch.Tensor]):
        if isinstance(losses, torch.Tensor):
            total_loss = losses
        else:
            total_loss = torch.stack(tuple(losses)).sum()
        total_loss.backward()

    @staticmethod
    def suggest_hyperparameters(name, trial, config):
        return dict()


class PCGrad(MultiTaskStrategy):
    """
    Implementation of PCGrad

    Paper:
        Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient surgery for multi-task
        learning. Advances in neural information processing systems, 33, 5824-5836.
    """

    __slots__ = ()

    def backward(self, *, losses: Sequence[torch.Tensor]):
        """Following 'Algorithm 1: PCGrad Update Rule' in the paper"""
        num_tasks = len(losses)
        trainable_params = [params for params in self._model.parameters() if params.requires_grad]

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

    @staticmethod
    def suggest_hyperparameters(name, trial, config):
        return dict()


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

    def __init__(self, optimiser, model, *, alpha, learning_rate):
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
        super().__init__(optimiser=optimiser, model=model)

        self._alpha = alpha
        self._learning_rate = learning_rate
        self._initial_losses = None
        self._loss_weights = None  # Will be initialised as ones in the first pass

    def to(self, device):
        if self._loss_weights is not None:
            self._loss_weights = self._loss_weights.to(device)

    def backward(self, losses: Sequence[torch.Tensor]):
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
        _gradient_norms = []  # G_W^{(i)} in the algorithm
        for loss in weighted_losses:
            self.zero_grad()
            loss.backward(retain_graph=True)
            _grads = torch.autograd.grad(outputs=loss, inputs=self._model.gradnorm_parameters(), retain_graph=True,
                                         create_graph=True)  # Needed to allow backprop through gradnorm_loss
            grads = torch.cat([grad.view(-1) for grad in _grads])
            _gradient_norms.append(torch.norm(grads))
        gradient_norms = torch.stack(_gradient_norms)

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
            self._loss_weights.data.clamp_(min=1e-6)  # Weights should be positive

            # Re-normalisation
            self._loss_weights.data *= num_tasks / self._loss_weights.data.sum()

    @staticmethod
    def suggest_hyperparameters(name, trial, config):
        return {"alpha": trial.suggest_float(f"{name}_alpha", **config["alpha"]),
                "learning_rate": trial.suggest_float(f"{name}_learning_rate", **config["learning_rate"])}


class UncertaintyWeighting(MultiTaskStrategy):
    """
    Implementation of Uncertainty Weighting for multi-task learning

    Paper:
        Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene
        geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition
        (pp. 7482-7491).
    """

    __slots__ = ("_log_vars",)

    def __init__(self, optimiser: optim.Optimizer, model: nn.Module):
        super().__init__(optimiser=optimiser, model=model)

        # Create a learnable log sigma^2 per task. These will be added to optimiser at first backpropagation
        self._log_vars: Optional[nn.Parameter] = None

    def backward(self, *, losses: Sequence[torch.Tensor]):
        device = losses[0].device
        num_tasks = len(losses)

        # Maybe initialise uncertainty parameters and add to optimiser
        if self._log_vars is None:
            self._log_vars = torch.nn.Parameter(torch.zeros(num_tasks, device=device), requires_grad=True)
            self._optimiser.add_param_group({"params": [self._log_vars]})

        # Compute loss
        total_loss = torch.stack(
            [0.5 * torch.exp(-log_var) * loss + 0.5 * log_var for log_var, loss in zip(self._log_vars, losses)]
        ).sum()

        # Compute gradients
        self.zero_grad()
        total_loss.backward()

    @staticmethod
    def suggest_hyperparameters(name, trial, config):
        return dict()


class MGDA(MultiTaskStrategy):
    """
    Implementation of MGDA algorithm for multi-task learning.

    Paper:
        Sener, O., & Koltun, V. (2018). Multi-task learning as multi-objective optimization. Advances in neural
        information processing systems, 31.

    Notes:
        This implementation uses the approach described in Algorithm 2 (Section 3.2) of the paper, but follows the
        original GitHub implementation which uses projected gradient descent instead of the Frank-Wolfe algorithm.

        The model must implement .shared_parameters(), which returns an iterator over the shared parameters across tasks
    """

    __slots__ = ()

    def backward(self, *, losses: Sequence[torch.Tensor]):
        """Using 'Algorithm 2 Update Equations for MTL') in the paper. Note that the model must implement
        .shared_parameters() which provides the parameters which are shared across the tasks"""
        # -------------
        # Compute gradients for all tasks (steps)
        # -------------
        shared_gradients = []
        for loss in losses:
            self.zero_grad()
            loss.backward(retain_graph=True)
            grads = torch.cat([params.grad.view(-1) for params in self._model.shared_parameters()
                               if params.grad is not None])
            shared_gradients.append(grads.detach().clone())

        # --------------
        # Solve the minimum-norm problem
        # --------------
        # Only two tasks here
        alphas = MinNormSolver.find_min_norm_element_two_task(shared_gradients)[0]

        # --------------
        # Compute loss and gradients
        # --------------
        total_loss = torch.stack([alpha * loss for alpha, loss in zip(alphas, losses)]).sum()

        # Backpropagation with combined loss
        self.zero_grad()
        total_loss.backward()

    @staticmethod
    def suggest_hyperparameters(name, trial, config):
        return dict()


# ---------------
# Functions
# ---------------
def get_available_mtl_strategies() -> Tuple[Type[MultiTaskStrategy], ...]:
    """All MTL strategies should be available from here"""
    return EqualWeighting, PCGrad, GradNorm, UncertaintyWeighting, MGDA


def get_mtl_strategy_type(name):
    """
    Function for getting a specified MTL strategy class

    Parameters
    ----------
    name : str

    Returns
    -------
    MultiTaskStrategy

    Examples
    --------
    >>> _ = get_mtl_strategy_type("GradNorm")
    >>> _ = get_mtl_strategy_type("PCGrad")
    >>> _ = get_mtl_strategy_type("EqualWeighting")
    >>> _ = get_mtl_strategy_type("UncertaintyWeighting")
    >>> _ = get_mtl_strategy_type("MGDA")
    >>> _ = get_mtl_strategy_type("NotAnMTLStrategy")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The MTL strategy 'NotAnMTLStrategy' was not recognised. Please select among the following: (...)
    """
    # Get all available strategies
    available_strategies = get_available_mtl_strategies()

    # Loop through and select the correct one
    for strategy in available_strategies:
        if name == strategy.__name__:
            return strategy

    # If no match, an error is raised
    raise ValueError(f"The MTL strategy {name!r} was not recognised. Please select among the following: "
                     f"{tuple(strategy.__name__ for strategy in available_strategies)}")


def suggest_multi_task_strategy(name, trial, config):
    name_prefix = "" if name is None else f"{name}_"

    strategy = trial.suggest_categorical(f"{name_prefix}strategy", choices=config.keys())

    strategy_name = name if name is None else f"{name}_{strategy}"
    hpcs = get_mtl_strategy_type(strategy).suggest_hyperparameters(strategy_name, trial=trial, config=config[strategy])
    return {"name": strategy, "kwargs": hpcs}


def get_mtl_strategy(name, *, optimiser, model, **kwargs):
    """
    Function for getting a specified MTL strategy

    Parameters
    ----------
    name : str
    optimiser : optim.Optimizer
    model : nn.Module
    kwargs

    Returns
    -------
    MultiTaskStrategy

    Examples
    --------
    >>> my_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
    >>> my_optimiser = optim.Adam(my_model.parameters(), lr=1e-4, betas=(0.6, 0.7))
    >>> _ = get_mtl_strategy("GradNorm", optimiser=my_optimiser, model=my_model, alpha=1.5, learning_rate=0.025)
    >>> _ = get_mtl_strategy("PCGrad", optimiser=my_optimiser, model=my_model)
    >>> _ = get_mtl_strategy("EqualWeighting", optimiser=my_optimiser, model=my_model)
    >>> _ = get_mtl_strategy("UncertaintyWeighting", optimiser=my_optimiser, model=my_model)
    >>> _ = get_mtl_strategy("MGDA", optimiser=my_optimiser, model=my_model)
    >>> _ = get_mtl_strategy("NotAnMTLStrategy", optimiser=my_optimiser, model=my_model)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: The MTL strategy 'NotAnMTLStrategy' was not recognised. Please select among the following: (...)
    """
    # Get all available strategies
    available_strategies = get_available_mtl_strategies()

    # Loop through and select the correct one
    for strategy in available_strategies:
        if name == strategy.__name__:
            return strategy(optimiser=optimiser, model=model, **kwargs)

    # If no match, an error is raised
    raise ValueError(f"The MTL strategy {name!r} was not recognised. Please select among the following: "
                     f"{tuple(strategy.__name__ for strategy in available_strategies)}")
