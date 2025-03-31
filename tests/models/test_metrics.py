import math
import random

import pytest
import torch

from elecssl.models.metrics import Histories, NaNValueError, MismatchClassCountError


# ----------------
# NaN in predictions
# ----------------
def test_nan_predictions_error_regression_metrics():
    """Test if NaNPredictionError is either correctly raised or no error at all, when there are NaN values in the model
    predictions, when computing regression scores"""
    # Create an invalid predicted tensor (containing NaN) and a valid ground truth tensor
    pred_tensor = torch.rand(size=(10, 1))
    pred_tensor[5] = float("nan")
    ground_truth = torch.rand(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_regression_metrics():
        try:
            result = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)
            assert math.isnan(result) and metric in ("conc_cc", "pearson_r", "spearman_rho")
        except NaNValueError:
            # If NaNValueError is raised, it's expected behavior for most metrics
            pass


def test_nan_predictions_error_classification_metrics():
    """Test if NaNPredictionError is correctly raised when there are NaN values in the model predictions, when computing
    classification scores"""
    # Create an invalid predicted tensor (containing NaN) and a valid ground truth tensor
    pred_tensor = torch.rand(size=(10, 1))
    pred_tensor[5] = float("nan")
    ground_truth = torch.rand(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_classification_metrics():
        try:
            result = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)
            assert math.isnan(result)
        except NaNValueError:
            # If NaNValueError is raised, it's expected behavior for most metrics
            pass


def test_nan_predictions_error_multiclass_classification_metrics():
    """For multiclass classification score, NaNPredictionError is sometimes raised, and other times it seems to be
    ignored"""
    # Create an invalid predicted tensor (containing NaN) and a valid ground truth tensor
    pred_tensor = torch.rand(size=(40, 5), dtype=torch.float) * 7
    pred_tensor[5] = float("nan")
    ground_truth = torch.tensor([random.randint(0, 4) for _ in range(40)])

    # Test metrics
    for metric in Histories.get_available_multiclass_classification_metrics():
        try:
            result = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)
            assert isinstance(result, float)
        except NaNValueError:
            # If NaNValueError is raised, it's expected behavior for many metrics
            pass


# ----------------
# NaN in ground truth
# ----------------
def test_nan_target_error_regression_metrics():
    """Test if NaNValueError is raised when there are NaN values in the ground truth, when computing regression
    scores"""
    # Create invalid predicted tensor and valid ground truth tensor
    pred_tensor = torch.rand(size=(10, 1))
    ground_truth = torch.rand(size=(10, 1))
    ground_truth[5] = float("nan")

    # Test metrics
    for metric in Histories.get_available_regression_metrics():
        try:
            result = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)
            assert math.isnan(result) and metric in ("conc_cc", "pearson_r", "spearman_rho")
        except NaNValueError:
            # If NaNValueError is raised, it's expected behavior for most metrics
            pass


def test_nan_target_error_classification_metrics():
    """Test if NaNValueError is raised when there are NaN values in the ground truth, when computing classification
    scores"""
    # Create invalid predicted tensor and valid ground truth tensor
    pred_tensor = torch.rand(size=(10, 1))
    ground_truth = torch.rand(size=(10, 1))
    ground_truth[5] = float("nan")

    # Test metrics
    for metric in Histories.get_available_classification_metrics():
        try:
            result = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)
            assert math.isnan(result)
        except NaNValueError:
            # If NaNValueError is raised, it's expected behavior for most metrics
            pass


def test_nan_target_error_multiclass_classification_metrics():
    """For multiclass classification score, NaNValueError or RuntimeError is raised"""
    # Create an invalid predicted tensor (containing NaN) and a valid ground truth tensor
    pred_tensor = torch.rand(size=(40, 5), dtype=torch.float) * 7
    ground_truth = torch.tensor([random.randint(0, 4) for _ in range(39)] + [float("nan")])

    # Test metrics
    for metric in Histories.get_available_multiclass_classification_metrics():
        with pytest.raises((NaNValueError, RuntimeError)):  # type: ignore
            Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)


# ----------------
# Constant predictions
# ----------------
def test_constant_predictions_regression_metrics():
    """Test if constant predictions does not raise any error messages"""
    # Create tensors
    pred_tensor = torch.ones(size=(10, 1))
    ground_truth = torch.rand(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_regression_metrics():
        _ = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)


def test_constant_predictions_classification_metrics():
    """Test if constant predictions does not raise any error messages"""
    # Create tensors
    pred_tensor = torch.ones(size=(10, 1))
    ground_truth = torch.rand(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_classification_metrics():
        _ = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)


def test_constant_predictions_multiclass_classification_metrics():
    """Test if constant predictions does not raise any error messages, except MismatchClassCountError"""
    # Create tensors
    pred_tensor = torch.rand(size=(40, 5), dtype=torch.float)
    ground_truth = torch.tensor([2] * 40, requires_grad=False, dtype=torch.int64)

    # Test metrics
    for metric in Histories.get_available_multiclass_classification_metrics():
        try:
            _ = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)
        except MismatchClassCountError:
            pass


# ----------------
# Constant targets
# ----------------
def test_constant_targets_regression_metrics():
    """Test if constant targets does not raise any error messages"""
    # Create tensors
    pred_tensor = torch.rand(size=(10, 1))
    ground_truth = torch.ones(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_regression_metrics():
        _ = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)


def test_constant_targets_classification_metrics():
    """Test if constant targets does not raise any error messages"""
    # Create tensors
    pred_tensor = torch.rand(size=(10, 1))
    ground_truth = torch.ones(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_classification_metrics():
        _ = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)


def test_constant_targets_multiclass_classification_metrics():
    """Test if constant targets does not raise any error messages"""
    # -------------
    # Version 1
    # -------------
    # Create tensors
    pred_tensor = torch.ones(size=(40, 5), dtype=torch.float)  # All ones
    ground_truth = torch.randint(size=(40,), low=0, high=5)

    # Test metrics
    for metric in Histories.get_available_multiclass_classification_metrics():
        _ = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)

    # -------------
    # Version 2
    # -------------
    # Create tensors
    pred_tensor = torch.zeros(size=(40, 5), dtype=torch.float)
    pred_tensor[:, 2] += 1  # 100% class number 2 (using zero indexing)
    ground_truth = torch.randint(size=(40,), low=0, high=5)

    # Test metrics
    for metric in Histories.get_available_multiclass_classification_metrics():
        _ = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)


# ----------------
# Inf in predictions are treated as nan
# ----------------
def test_inf_predictions_error_regression_metrics():
    """Test if NaNPredictionError is either correctly raised or no error at all, when there are inf values in the model
    predictions, when computing regression scores"""
    # Create an invalid predicted tensor (containing inf) and a valid ground truth tensor
    pred_tensor = torch.rand(size=(10, 1))
    pred_tensor[5] = float("inf")
    ground_truth = torch.rand(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_regression_metrics():
        try:
            Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)

            # Correlations can be numeric
            assert metric in ('conc_cc', 'pearson_r', 'spearman_rho'), (f"The metric {metric!r} did not raise "
                                                                        f"NaNValueError, which was unexpected")
        except NaNValueError:
            # If NaNValueError is raised, it's expected behavior for most metrics
            pass


def test_inf_predictions_error_classification_metrics():
    """Test if NaNPredictionError is either correctly raised or no error at all, when there are inf values in the model
    predictions, when computing classification scores"""
    # Create an invalid predicted tensor (containing inf) and a valid ground truth tensor
    pred_tensor = torch.rand(size=(10, 1))
    pred_tensor[5] = float("inf")
    ground_truth = torch.rand(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_classification_metrics():
        try:
            result = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)
            assert metric in ("auc",) and math.isnan(result), (f"The metric {metric!r} did not raise NaNValueError, "
                                                               f"and obtained an unexpected value {result}")
        except NaNValueError:
            # If NaNValueError is raised, it's expected behavior for most metrics
            pass


def test_inf_predictions_error_multiclass_classification_metrics():
    """For multiclass classification score, NaNPredictionError is sometimes raised, and other times it seems to be
    ignored"""
    # Create an invalid predicted tensor (containing inf) and a valid ground truth tensor
    pred_tensor = torch.rand(size=(40, 5), dtype=torch.float) * 7
    pred_tensor[5] = float("inf")
    ground_truth = torch.tensor([random.randint(0, 4) for _ in range(40)])

    # Test metrics
    for metric in Histories.get_available_multiclass_classification_metrics():
        try:
            result = Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)
            assert isinstance(result, float)
        except NaNValueError:
            # If NaNValueError is raised, it's expected behavior for many metrics
            pass
