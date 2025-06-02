import copy

import pytest
import torch
from torch import nn, optim

from elecssl.models.losses import get_pytorch_loss_function
from elecssl.models.mtl_strategies.multi_task_strategies import PCGrad, GradNorm, UncertaintyWeighting, MGDA


class _Model(nn.Module):
    """
    A dummy model which uses the residual of one task as the input to the second.

    Would be cool to try this for age prediction too. Finding the maximally relevant brain age prediction models

    Examples
    --------
    >>> _Model(10)
    _Model(
      (_main_model): Sequential(
        (0): Linear(in_features=10, out_features=5, bias=True)
        (1): ReLU()
        (2): Linear(in_features=5, out_features=1, bias=True)
      )
      (_residual_model): Linear(in_features=1, out_features=1, bias=True)
    )
    >>> for my_param in _Model(10).gradnorm_parameters():
    ...     my_param
    Parameter containing:
    tensor([[...]], requires_grad=True)
    >>> for my_param in _Model(10).shared_parameters():
    ...     my_param
    ...     my_param.data.size()
    Parameter containing:
    tensor([[...]], requires_grad=True)
    torch.Size([5, 10])
    Parameter containing:
    tensor([...], requires_grad=True)
    torch.Size([5])
    Parameter containing:
    tensor([...], requires_grad=True)
    torch.Size([1, 5])
    Parameter containing:
    tensor([...], requires_grad=True)
    torch.Size([1])
    """

    def __init__(self, in_features: int):
        super().__init__()

        self._main_model = nn.Sequential(
            nn.Linear(in_features, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        self._residual_model = nn.Linear(1, 1, bias=True)  # Predict from the residual

    def forward(self, x, *, pretext_y):
        # Predict on the pretext task
        pretext_prediction = self._main_model(x)

        # Make prediction from residual
        residuals = pretext_prediction - pretext_y
        downstream_prediction = self._residual_model(residuals)

        return pretext_prediction, downstream_prediction

    # ------------
    # Additional requirements
    # ------------
    def gradnorm_parameters(self):
        """This is required for GradNorm"""
        yield next(self._main_model[-1].parameters())

    def shared_parameters(self):
        """This is required for MGDA"""
        for param in self._main_model.parameters():
            yield param


def _create_model(in_features, device):
    model = _Model(in_features=in_features)
    model.train()
    return model.to(device)


# ------------
# Tests for PCGrad
# ------------
@pytest.mark.parametrize("in_features,loss,learning_rate,device_name", [
    (1, "L1Loss", 1e-5, "cuda"), (12, "MSELoss", 2.3e-3, "cpu"), (10, "L1Loss", 1e1, "cuda"),
    (1, "L1Loss", 1e-2, "cpu"), (1, "MSELoss", 1e-5, "cpu")
])
def test_pcgrad(in_features, loss, learning_rate, device_name):
    """Test if using PCGrad works"""
    batch_size = 10
    num_dummy_epochs = 7

    # Skip if device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    # Dummy model
    model = _create_model(in_features, device)

    # Dummy data
    x_batches = (torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # Loss function
    criterion = get_pytorch_loss_function(loss).to(device)

    # ---------------
    # Forward passes for two tasks
    # ---------------
    # PCGrad setup
    base_optim = optim.Adam(model.parameters(), lr=learning_rate)
    strategy = PCGrad(base_optim)
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute losses
        pretext_yhat, downstream_yhat = model(x, pretext_y=pretext_y)
        loss1 = criterion(pretext_yhat, pretext_y)
        loss2 = criterion(downstream_yhat, downstream_y)

        # Backward using PCGrad
        strategy.zero_grad()
        strategy.backward(losses=[loss1, loss2], model=model)

        # Check gradients have been populated
        grads_present = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert all(grads_present), "Some parameters are missing gradients after backpropagation."

        # Step
        strategy.step()


@pytest.mark.parametrize("in_features,loss,learning_rate,device_name", [
    (1, "L1Loss", 1e-5, "cuda"), (12, "MSELoss", 2.3e-3, "cpu"), (10, "L1Loss", 1e-1, "cuda"),
    (1, "L1Loss", 1e-2, "cpu"), (1, "MSELoss", 1e-5, "cpu")
])
def test_pcgrad_preserves_frozen_layers_and_updates_others(in_features, loss, learning_rate, device_name):
    """Test if PCGrad implementation preserves the frozen layers and updates the others"""
    batch_size = 10
    num_dummy_epochs = 30

    # Skip if device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    # Dummy model
    model = _create_model(in_features, device)

    # Freeze the first layer
    model._main_model[0].weight.requires_grad = False
    model._main_model[0].bias.requires_grad = False

    # Save the original state
    original_state = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})

    # Dummy data
    x_batches = (torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # Loss function
    criterion = get_pytorch_loss_function(loss).to(device)

    # ---------------
    # Training
    # ---------------
    # Optimizer and PCGrad
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    pcgrad = PCGrad(optimizer)
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute losses
        pretext_yhat, downstream_yhat = model(x, pretext_y=pretext_y)
        loss1 = criterion(pretext_yhat, pretext_y)
        loss2 = criterion(downstream_yhat, downstream_y)

        pcgrad.zero_grad()
        pcgrad.backward(losses=[loss1, loss2], model=model)
        pcgrad.step()

        # Check frozen layer gradients are None
        assert model._main_model[0].weight.grad is None, "Frozen weight has a gradient!"
        assert model._main_model[0].bias.grad is None, "Frozen bias has a gradient!"

    # ---------------
    # Tests
    # ---------------
    # Check that frozen parameters are the same, and that non-frozen parameters changed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(param.data, original_state[name]), \
                f"Trainable parameters ({name!r}) were not updated!"
        else:
            assert torch.allclose(param.data, original_state[name]), f"Frozen parameters ({name!r}) were updated!"


@pytest.mark.parametrize("seed,loss_name,learning_rate,device_name,in_features", [
    (0, "L1Loss", 1e-5, "cuda", 16), (12, "MSELoss", 2.3e-3, "cpu", 4), (10, "L1Loss", 3, "cuda", 2),
    (7, "L1Loss", 1e-2, "cpu", 32), (7, "MSELoss", 1e-5, "cpu", 15)
])
def test_pcgrad_equivalent_to_adam_on_identical_losses(seed, loss_name, learning_rate, device_name, in_features):
    """When the two losses are the same, PCGrad should behave identically to vanilla Adam (up to numerical precision).
    This is because the gradients should never be conflicting, hence gradient surgery is never actually applied."""
    torch.manual_seed(seed)
    batch_size = 10
    num_dummy_epochs = 30

    # Skip if the device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    model_1 = _create_model(in_features=in_features, device=device)
    model_2 = _create_model(in_features=in_features, device=device)
    model_2.load_state_dict(copy.deepcopy(model_1.state_dict()))  # Clone weights
    model_3 = _create_model(in_features=in_features, device=device)
    model_3.load_state_dict(copy.deepcopy(model_1.state_dict()))  # Clone weights

    # Dummy data
    x_batches = tuple(torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = tuple(torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = tuple(torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # ---------------
    # Adam baseline
    # ---------------
    # Loss function and baseline optimiser
    criterion = get_pytorch_loss_function(loss_name).to(device)
    opt_adam = torch.optim.Adam(model_1.parameters(), lr=learning_rate)

    # Training
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute loss, gradients, and perform step
        pretext_yhat, downstream_yhat = model_1(x, pretext_y=pretext_y)
        loss = 2 * criterion(downstream_yhat, downstream_y)  # Downstream only. Need to multiply with two to simulate
        # two tasks

        # Adam baseline
        opt_adam.zero_grad()
        loss.backward()
        opt_adam.step()

    # ---------------
    # PCGrad with duplicate loss
    # ---------------
    # Create MTL strategy
    criterion = get_pytorch_loss_function(loss_name).to(device)
    opt_pc = optim.Adam(model_2.parameters(), lr=learning_rate)
    pcgrad = PCGrad(opt_pc)

    # Training
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute loss, gradients, and perform step
        pretext_yhat, downstream_yhat = model_2(x, pretext_y=pretext_y)
        loss = criterion(downstream_yhat, downstream_y)  # Downstream only

        # PCGrad
        pcgrad.zero_grad()
        pcgrad.backward(losses=[loss, loss], model=model_2)
        pcgrad.step()

    # Compare weights and gradients
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        assert torch.allclose(p1.data, p2.data), f"Parameters are not the same\n{p1.data}\n{p2.data}"
        assert torch.allclose(p1.grad, p2.grad), f"Gradients are not the same\n{p1.grad}\n{p2.grad}"


# ------------
# Tests for GradNorm
# ------------
@pytest.mark.parametrize("alpha,gradnorm_lr,in_features,loss,learning_rate,device_name", [
    (1.5, 0.9, 1, "L1Loss", 1e-5, "cuda"), (1.2, 0.1, 12, "MSELoss", 2.3e-3, "cpu"),
    (0, 0.04, 10, "L1Loss", 1e1, "cuda"), (0.771, 0.05, 1, "L1Loss", 1e-2, "cpu"),
    (3, 0.0001, 1, "MSELoss", 1e-5, "cpu")
])
def test_grad_norm(alpha, gradnorm_lr, in_features, loss, learning_rate, device_name):
    """Test if using GradNorm works"""
    batch_size = 10
    num_dummy_epochs = 7

    # Skip if device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    # Dummy model
    model = _create_model(in_features, device)

    # Dummy data
    x_batches = (torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # Loss function
    criterion = get_pytorch_loss_function(loss).to(device)

    # ---------------
    # Training with GradNorm
    # ---------------
    # GradNorm setup
    base_optim = optim.Adam(model.parameters(), lr=learning_rate)
    strategy = GradNorm(base_optim, alpha=alpha, learning_rate=gradnorm_lr)
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute losses
        pretext_yhat, downstream_yhat = model(x, pretext_y=pretext_y)
        loss1 = criterion(pretext_yhat, pretext_y)
        loss2 = criterion(downstream_yhat, downstream_y)

        # Backward
        strategy.zero_grad()
        strategy.backward(losses=[loss1, loss2], model=model)

        # Check gradients have been populated
        grads_present = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert all(grads_present), "Some parameters are missing gradients after backpropagation."

        # Step
        strategy.step()


@pytest.mark.parametrize("alpha,gradnorm_lr,in_features,loss,learning_rate,device_name", [
    (1.5, 0.9, 1, "L1Loss", 1e-5, "cuda"), (1.2, 0.1, 12, "MSELoss", 2.3e-3, "cpu"),
    (0, 0.04, 10, "L1Loss", 1e1, "cuda"), (0.771, 0.05, 1, "L1Loss", 1e-2, "cpu"),
    (3, 0.0001, 1, "MSELoss", 1e-5, "cpu")
])
def test_gradnorm_weights_sign_and_sum(alpha, gradnorm_lr, in_features, loss, learning_rate, device_name):
    """Test if the weights are always positive and that they sum to the number of tasks"""
    batch_size = 10
    num_dummy_epochs = 7

    # Skip if device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    # Dummy model
    model = _create_model(in_features, device)

    # Dummy data
    x_batches = (torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # Loss function
    criterion = get_pytorch_loss_function(loss).to(device)

    # ---------------
    # Training
    # ---------------
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    strategy = GradNorm(optimizer, alpha=alpha, learning_rate=gradnorm_lr)
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute losses
        pretext_yhat, downstream_yhat = model(x, pretext_y=pretext_y)
        loss1 = criterion(pretext_yhat, pretext_y)
        loss2 = criterion(downstream_yhat, downstream_y)
        losses = [loss1, loss2]

        # Backward and update
        strategy.zero_grad()
        strategy.backward(losses=losses, model=model)
        strategy.step()

        # Test
        assert (strategy._loss_weights.data > 0).all(), "Loss weights should remain positive"
        assert abs(strategy._loss_weights.data.sum().item() - len(losses)) < 1e-4, \
            "Loss weights should sum to number of tasks"


@pytest.mark.parametrize("alpha,gradnorm_lr,in_features,loss,learning_rate,device_name", [
    (1.5, 0.9, 1, "L1Loss", 1e-5, "cuda"), (1.2, 0.1, 12, "MSELoss", 2.3e-3, "cpu"),
    (0, 0.04, 10, "L1Loss", 1e1, "cuda"), (0.771, 0.05, 1, "L1Loss", 1e-2, "cpu"),
    (3, 0.0001, 1, "MSELoss", 1e-5, "cpu")
])
def test_gradnorm_preserves_frozen_layers_and_updates_others(alpha, gradnorm_lr, in_features, loss, learning_rate,
                                                             device_name):
    """Test if GradNorm implementation preserves the frozen layers and updates the others"""
    batch_size = 10
    num_dummy_epochs = 30

    # Skip if device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    # Dummy model
    model = _create_model(in_features, device)

    # Freeze the first layer
    model._main_model[0].weight.requires_grad = False
    model._main_model[0].bias.requires_grad = False

    # Save the original state
    original_state = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})

    # Dummy data
    x_batches = (torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # Loss function
    criterion = get_pytorch_loss_function(loss).to(device)

    # ---------------
    # Training
    # ---------------
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    strategy = GradNorm(optimizer, alpha=alpha, learning_rate=gradnorm_lr)
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute losses
        pretext_yhat, downstream_yhat = model(x, pretext_y=pretext_y)
        loss1 = criterion(pretext_yhat, pretext_y)
        loss2 = criterion(downstream_yhat, downstream_y)

        strategy.zero_grad()
        strategy.backward(losses=[loss1, loss2], model=model)
        strategy.step()

        # Check frozen layer gradients are None
        assert model._main_model[0].weight.grad is None, "Frozen weight has a gradient!"
        assert model._main_model[0].bias.grad is None, "Frozen bias has a gradient!"

    # ---------------
    # Tests
    # ---------------
    # Check that frozen parameters are the same, and that non-frozen parameters changed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(param.data, original_state[name]), \
                f"Trainable parameters ({name!r}) were not updated!"
        else:
            assert torch.allclose(param.data, original_state[name]), f"Frozen parameters ({name!r}) were updated!"


# ------------
# Tests for UncertaintyWeighting
# ------------
@pytest.mark.parametrize("in_features,loss,learning_rate,device_name", [
    (1, "L1Loss", 1e-5, "cuda"), (12, "MSELoss", 2.3e-3, "cpu"), (10, "L1Loss", 1e1, "cuda"),
    (1, "L1Loss", 1e-2, "cpu"), (1, "MSELoss", 1e-5, "cpu")
])
def test_uncertainty_weighting(in_features, loss, learning_rate, device_name):
    """Test of the UncertaintyWeighting works for optimisation"""
    batch_size = 10
    num_dummy_epochs = 7

    # Skip if device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    # Dummy model
    model = _create_model(in_features, device)

    # Dummy data
    x_batches = (torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # Loss function
    criterion = get_pytorch_loss_function(loss).to(device)

    # ---------------
    # Training with Uncertainty Weighting
    # ---------------
    # Uncertainty Weighting setup
    base_optim = optim.Adam(model.parameters(), lr=learning_rate)
    strategy = UncertaintyWeighting(base_optim)
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute losses
        pretext_yhat, downstream_yhat = model(x, pretext_y=pretext_y)
        loss1 = criterion(pretext_yhat, pretext_y)
        loss2 = criterion(downstream_yhat, downstream_y)

        # Backward
        strategy.zero_grad()
        strategy.backward(losses=[loss1, loss2], model=model)

        # Check gradients have been populated
        grads_present = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert all(grads_present), "Some parameters are missing gradients after backpropagation."

        # Check that log_vars are now in one of the param groups
        assert any(strategy._log_vars is p for group in base_optim.param_groups for p in group['params']), \
            "log_vars were not added to the optimizer"

        # Step
        strategy.step()


@pytest.mark.parametrize("in_features,loss,learning_rate,device_name", [
    (1, "L1Loss", 1e-5, "cuda"), (12, "MSELoss", 2.3e-3, "cpu"), (10, "L1Loss", 1e1, "cuda"),
    (1, "L1Loss", 1e-2, "cpu"), (1, "MSELoss", 1e-5, "cpu")
])
def test_uncertainty_weighting_preserves_frozen_layers_and_updates_others(in_features, loss, learning_rate,
                                                                          device_name):
    """Test if UncertaintyWeighting implementation preserves the frozen layers and updates the others"""
    batch_size = 10
    num_dummy_epochs = 30

    # Skip if device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    # Dummy model
    model = _create_model(in_features, device)

    # Freeze the first layer
    model._main_model[0].weight.requires_grad = False
    model._main_model[0].bias.requires_grad = False

    # Save the original state
    original_state = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})

    # Dummy data
    x_batches = (torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # Loss function
    criterion = get_pytorch_loss_function(loss).to(device)

    # ---------------
    # Training
    # ---------------
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    strategy = UncertaintyWeighting(optimizer)
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute losses
        pretext_yhat, downstream_yhat = model(x, pretext_y=pretext_y)
        loss1 = criterion(pretext_yhat, pretext_y)
        loss2 = criterion(downstream_yhat, downstream_y)

        strategy.zero_grad()
        strategy.backward(losses=[loss1, loss2], model=model)
        strategy.step()

        # Check frozen layer gradients are None
        assert model._main_model[0].weight.grad is None, "Frozen weight has a gradient!"
        assert model._main_model[0].bias.grad is None, "Frozen bias has a gradient!"

    # ---------------
    # Tests
    # ---------------
    # Check that frozen parameters are the same, and that non-frozen parameters changed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(param.data, original_state[name]), \
                f"Trainable parameters ({name!r}) were not updated!"
        else:
            assert torch.allclose(param.data, original_state[name]), f"Frozen parameters ({name!r}) were updated!"


# ------------
# Tests for MGDA
# ------------
@pytest.mark.parametrize("in_features,loss,learning_rate,device_name", [
    (1, "L1Loss", 1e-5, "cuda"), (12, "MSELoss", 2.3e-3, "cpu"), (10, "L1Loss", 1e1, "cuda"),
    (1, "L1Loss", 1e-2, "cpu"), (1, "MSELoss", 1e-5, "cpu")
])
def test_mgda(in_features, loss, learning_rate, device_name):
    """Test if using MGDA works"""
    batch_size = 10
    num_dummy_epochs = 7

    # Skip if device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    # Dummy model
    model = _create_model(in_features, device)

    # Dummy data
    x_batches = (torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # Loss function
    criterion = get_pytorch_loss_function(loss).to(device)

    # ---------------
    # Training with GradNorm
    # ---------------
    # MGDA setup
    base_optim = optim.Adam(model.parameters(), lr=learning_rate)
    strategy = MGDA(base_optim)
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute losses
        pretext_yhat, downstream_yhat = model(x, pretext_y=pretext_y)
        loss1 = criterion(pretext_yhat, pretext_y)
        loss2 = criterion(downstream_yhat, downstream_y)

        # Backward
        strategy.zero_grad()
        strategy.backward(losses=[loss1, loss2], model=model)

        # Check gradients have been populated
        grads_present = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert all(grads_present), "Some parameters are missing gradients after backpropagation."

        # Step
        strategy.step()


@pytest.mark.parametrize("in_features,loss,learning_rate,device_name", [
    (1, "L1Loss", 1e-5, "cuda"), (12, "MSELoss", 2.3e-3, "cpu"), (10, "L1Loss", 1e1, "cuda"),
    (1, "L1Loss", 1e-2, "cpu"), (1, "MSELoss", 1e-5, "cpu")
])
def test_mgda_preserves_frozen_layers_and_updates_others(in_features, loss, learning_rate, device_name):
    """Test if MGDA implementation preserves the frozen layers and updates the others"""
    batch_size = 10
    num_dummy_epochs = 30

    # Skip if device does not exist
    device = torch.device(device_name)
    try:
        torch.tensor([0.0]).to(device)
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")

    # ---------------
    # Fix dummy model, data, and loss
    # ---------------
    # Dummy model
    model = _create_model(in_features, device)

    # Freeze the first layer
    model._main_model[0].weight.requires_grad = False
    model._main_model[0].bias.requires_grad = False

    # Save the original state
    original_state = copy.deepcopy({k: v.clone() for k, v in model.state_dict().items()})

    # Dummy data
    x_batches = (torch.rand(batch_size, in_features) for _ in range(num_dummy_epochs))
    pretext_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))
    downstream_y_batches = (torch.rand(batch_size, 1) for _ in range(num_dummy_epochs))

    # Loss function
    criterion = get_pytorch_loss_function(loss).to(device)

    # ---------------
    # Training
    # ---------------
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    strategy = MGDA(optimizer)
    for x, pretext_y, downstream_y in zip(x_batches, pretext_y_batches, downstream_y_batches):
        x = x.to(device)
        pretext_y = pretext_y.to(device)
        downstream_y = downstream_y.to(device)

        # Compute losses
        pretext_yhat, downstream_yhat = model(x, pretext_y=pretext_y)
        loss1 = criterion(pretext_yhat, pretext_y)
        loss2 = criterion(downstream_yhat, downstream_y)

        strategy.zero_grad()
        strategy.backward(losses=[loss1, loss2], model=model)
        strategy.step()

        # Check frozen layer gradients are None
        assert model._main_model[0].weight.grad is None, "Frozen weight has a gradient!"
        assert model._main_model[0].bias.grad is None, "Frozen bias has a gradient!"

    # ---------------
    # Tests
    # ---------------
    # Check that frozen parameters are the same, and that non-frozen parameters changed
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.allclose(param.data, original_state[name]), \
                f"Trainable parameters ({name!r}) were not updated!"
        else:
            assert torch.allclose(param.data, original_state[name]), f"Frozen parameters ({name!r}) were updated!"
