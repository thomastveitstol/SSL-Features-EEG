import pytest
import torch

from elecssl.models.mtl_strategies.minimum_norm_solver import MinNormSolver


@pytest.mark.parametrize("grad_dimension,device_name", (
        (5, "cuda"), (3, "cpu"), (2, "cpu"), (90, "cuda"), (76, "cuda"), (34, "cuda"), (50, "cpu"), (86, "cpu"),
        (34, "cpu"), (20, "cuda"), (55, "cuda"), (129, "cpu"), (7, "cpu"), (11, "cuda"), (22, "cpu"), (62, "cuda"),
))
def test_two_task_solution(grad_dimension, device_name):
    """Test if the two-task implementation gives the same alpha values as multi-task learning with number of
    tasks = 2"""
    solver = MinNormSolver

    # Create gradients (which are just tensors)
    try:
        tensor_1 = torch.rand(size=(grad_dimension,))
        tensor_2 = torch.rand(size=(grad_dimension,))
    except (AssertionError, RuntimeError, ValueError) as e:
        pytest.skip(f"Skipping test on {device_name!r}: {str(e)}")
        tensor_1 = None
        tensor_2 = None

    assert tensor_1 is not None  # mypy complains
    assert tensor_2 is not None

    # Compute alpha values
    alphas_generic = torch.tensor(solver.find_min_norm_element((tensor_1, tensor_2))[0], dtype=torch.float)
    alphas_two_task = solver.find_min_norm_element_two_task((tensor_1, tensor_2))[0]

    # Tests
    assert torch.allclose(alphas_generic, alphas_two_task, rtol=1e-3, atol=1e-3), \
        f"The two solvers did not produce the same alpha values: {alphas_generic}, {alphas_two_task}"
