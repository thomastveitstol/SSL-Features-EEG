import math
import random

import pytest
import torch

from elecssl.models.metrics import Histories, NaNValueError


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
            assert math.isnan(result)
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
            assert math.isnan(result)
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
