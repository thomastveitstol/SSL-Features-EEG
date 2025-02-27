import pytest
import torch

from elecssl.models.metrics import Histories, NaNPredictionError


def test_nan_predictions_error_regression_metrics():
    """Test if NaNPredictionError is correctly raised when there are NaN values in the model predictions, when computing
    regression scores"""
    # Create invalid predicted tensor and valid ground truth tensor
    pred_tensor = torch.rand(size=(10, 1))
    pred_tensor[5] = float("nan")
    ground_truth = torch.rand(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_regression_metrics():
        with pytest.raises(NaNPredictionError):
            Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)


def test_nan_predictions_error_classification_metrics():
    """Test if NaNPredictionError is correctly raised when there are NaN values in the model predictions, when computing
    classification scores"""
    # Create invalid predicted tensor and valid ground truth tensor
    pred_tensor = torch.rand(size=(10, 1))
    pred_tensor[5] = float("nan")
    ground_truth = torch.rand(size=(10, 1))

    # Test metrics
    for metric in Histories.get_available_classification_metrics():
        with pytest.raises(NaNPredictionError):
            Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)


def test_nan_predictions_error_multiclass_classification_metrics():
    """Test if NaNPredictionError is correctly raised when there are NaN values in the model predictions, when computing
    multiclass classification scores"""
    # Create invalid predicted tensor and valid ground truth tensor
    pred_tensor = torch.rand(size=(10, 5))
    pred_tensor[5] = float("nan")
    ground_truth = torch.rand(size=(10, 5))

    # Test metrics
    for metric in Histories.get_available_multiclass_classification_metrics():
        with pytest.raises(NaNPredictionError):
            Histories.compute_metric(metric=metric, y_pred=pred_tensor, y_true=ground_truth)
