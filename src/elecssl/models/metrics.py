"""
Implementing classification Histories class for storing training and validation metrics during training

There will likely be overlap with two other implementations at
https://github.com/thomastveitstol/RegionBasedPoolingEEG/blob/master/src/metrics.py and
https://github.com/thomastveitstol/CrossDatasetLearningEEG/blob/master/src/cdl_eeg/models/metrics.py

Author: Thomas Tveitstøl (Oslo University Hospital)
"""
# mypy: disable-error-code="index,union-attr"
import itertools
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any, NamedTuple, Union

import numpy
import pandas
import torch
from matplotlib import pyplot
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, roc_auc_score, \
    r2_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score, median_absolute_error, \
    explained_variance_score, max_error
from torch import nn

from elecssl.data.subject_split import Subject


# ----------------
# Convenient decorators
# ----------------
def regression_metric(is_higher_better):
    def decorator(func):
        setattr(func, "_is_regression_metric", True)
        setattr(func, "_higher_is_better", is_higher_better)
        return func
    return decorator


def classification_metric(is_higher_better):
    def decorator(func):
        setattr(func, "_is_classification_metric", True)
        setattr(func, "_higher_is_better", is_higher_better)
        return func
    return decorator


def multiclass_classification_metric(is_higher_better):
    def decorator(func):
        setattr(func, "_is_multiclass_classification_metric", True)
        setattr(func, "_higher_is_better", is_higher_better)
        return func
    return decorator


def groups_metric(func):
    setattr(func, "_is_groups_metric", True)
    return func


# ----------------
# Convenient small classes
# ----------------
class YYhat(NamedTuple):
    """Tuple for storing target and prediction"""
    y_true: torch.Tensor
    y_pred: torch.Tensor


# ----------------
# Classes
# ----------------
class Histories:
    """
    Class for keeping track of all metrics during training. Works for both classification and regression

    Keep in mind that for scores computed on the training dataset, it is strange to store outputs and targets, and
    compute metrics at the end of the epoch for correlation metrics in particular, as the outputs are computed with
    different weights. However, training curves are not sufficiently interesting to justify the time it would take to
    repeat the forward pass on the entire training set after each epoch

    Examples
    --------
    >>> Histories.get_available_classification_metrics()
    ('auc',)
    >>> Histories.get_available_regression_metrics()  # doctest: +NORMALIZE_WHITESPACE
    ('conc_cc', 'explained_variance', 'mae', 'mape', 'max_error', 'med_ae', 'mse', 'pearson_r', 'r2_score',
     'spearman_rho')
    >>> Histories.get_available_multiclass_classification_metrics()
    ('acc', 'auc_ovo', 'auc_ovr', 'balanced_acc', 'ce_loss', 'kappa', 'mcc')
    >>> Histories.get_available_metrics()  # doctest: +NORMALIZE_WHITESPACE
    ('auc', 'acc', 'auc_ovo', 'auc_ovr', 'balanced_acc', 'ce_loss', 'kappa', 'mcc', 'conc_cc', 'explained_variance',
     'mae', 'mape', 'max_error', 'med_ae', 'mse', 'pearson_r', 'r2_score', 'spearman_rho')
    """

    __slots__ = ("_history", "_prediction_history", "_subgroup_histories", "_epoch_y_pred", "_epoch_y_true",
                 "_epoch_subjects", "_name", "_variables_history", "_variable_metrics", "_variables_history_ratios",
                 "_groups_history_diff", "_groups_history_ratio")

    def __init__(self, metrics, *, name=None, splits, expected_variables=None, variable_metrics=None):
        """
        Initialise

        Parameters
        ----------
        metrics : str | tuple[str, ...]
            The metrics to use. If 'str', it must either be 'regression' or 'classification', specifying that all
            available regression/classification metrics should be used
        name : str, optional
            May be used for the printing of the metrics
        splits : dict[str, typing.Any] | None
            Splits for computing metrics per subgroup. Each 'split' must be an attribute of the Subject objects passed
            to 'store_batch_evaluation'. Note that this should also work for any class inheriting from Subject, allowing
            for more customised subgroup splitting
        expected_variables : dict[str, tuple[str, ...]] | None
        variable_metrics : dict[str, str | tuple[str, ...]] | None
        """
        # Maybe set metrics
        if metrics == "regression":
            metrics = self.get_available_regression_metrics()
        elif metrics == "classification":
            metrics = self.get_available_classification_metrics()
        elif metrics == "multiclass_classification":
            metrics = self.get_available_multiclass_classification_metrics()

        # ----------------
        # Input checks
        # ----------------
        # Check if all metrics are implemented
        if not all(metric in self.get_available_metrics() for metric in metrics):
            raise ValueError(f"The following metrics were not recognised: "
                             f"{set(metric for metric in metrics if metric not in self.get_available_metrics())}. The "
                             f"allowed ones are: {self.get_available_metrics()}")

        # Type check the name
        if name is not None and not isinstance(name, str):
            raise TypeError(f"Unrecognised name type: {type(name)}")

        # ----------------
        # Set attributes
        # ----------------
        self._name = name

        # ----------------
        # Create history dictionaries
        # ----------------
        # The "normal" one
        self._history: Dict[str, List[float]] = {f"{metric}": [] for metric in metrics}

        # For storing all predictions .
        self._prediction_history: Dict[Subject, List[List[Union[float, Tuple[float, ...]]]]] = dict()

        # Histories per subgroup. Not doing any check for legal split names, as the subjects may inherit from Subject
        # and thus have additional attributes
        # Could e.g. be {dataset_name: {D1: {metric1: [val1, val2, val3]}}}
        _sub_hist: Optional[Dict[str, Dict[Any, Dict[str, List[Union[float]]]]]]
        if splits is not None:
            _sub_hist = {split: {sub_group: {metric: [] for metric in metrics} for sub_group in sub_groups}
                         for split, sub_groups in splits.items()}
        else:
            _sub_hist = None
        self._subgroup_histories = _sub_hist

        # For storing variables such as age, ravlt_tot etc., which we will (e.g.) correlate with the difference between
        # y_pred and y_true. I think it'll be best to have variable name first, then the datasets. Should consider
        # cleaning up here, a little unreadable...
        self._variables_history: Optional[Dict[str, Dict[str, Dict[str, Union[Dict[str, List[float]], List[float]]]]]]
        self._variables_history_ratios: Optional[Dict[str, Dict[str, Dict[str, Union[Dict[str, List[float]],
                                                                                     List[float]]]]]]
        self._variable_metrics: Optional[Dict[str, Tuple[str, ...]]]
        if expected_variables is not None and any(variable_names for variable_names in expected_variables.values()):
            _metrics: Dict[str, Tuple[str, ...]] = {}
            for var_name, var_metrics in variable_metrics.items():
                if var_metrics == "regression":
                    _metrics[var_name] = self.get_available_regression_metrics()
                elif var_metrics == "classification":
                    _metrics[var_name] = self.get_available_classification_metrics()
                elif var_metrics == "multiclass_classification":
                    _metrics[var_name] = self.get_available_multiclass_classification_metrics()
                elif var_metrics == "groups_metrics":
                    _metrics[var_name] = self.get_available_groups_metrics()
                else:
                    _metrics[var_name] = var_metrics

            self._variables_history = {
                var_name: {dataset_name: {metric: {} if metric in self.get_available_groups_metrics() else []
                                          for metric in _metrics[var_name]}
                           for dataset_name in itertools.chain(("All",), dataset_names)}
                for var_name, dataset_names in expected_variables.items()
            }
            self._variables_history_ratios = {
                var_name: {dataset_name: {metric: {} if metric in self.get_available_groups_metrics() else []
                                          for metric in _metrics[var_name]}
                           for dataset_name in itertools.chain(("All",), dataset_names)}
                for var_name, dataset_names in expected_variables.items()
            }
            self._variable_metrics = _metrics
        else:
            self._variables_history = None
            self._variables_history_ratios = None
            self._variable_metrics = None

        # ----------------
        # Initialise epochs predictions and targets.
        # They will be updated for each batch
        # ----------------
        self._epoch_y_pred: List[torch.Tensor] = []
        self._epoch_y_true: List[torch.Tensor] = []
        self._epoch_subjects: List[Subject] = []

    def store_batch_evaluation(self, y_pred, y_true, subjects=None):
        """
        Store the prediction, targets, and maybe the corresponding subjects. Should be called for each batch

        Parameters
        ----------
        y_pred : torch.Tensor
        y_true : torch.Tensor
        subjects : tuple[Subject, ...], optional

        Returns
        -------
        None
        """
        self._epoch_y_pred.append(y_pred)
        self._epoch_y_true.append(y_true)

        # Store prediction in predictions history
        for prediction, subject in zip(y_pred, subjects):
            if subject in self._prediction_history:
                if prediction.size()[0] > 1:
                    _prediction = tuple(float(pred) for pred in prediction.cpu().tolist())
                elif prediction.size()[0] == 1:
                    _prediction = float(prediction)  # type: ignore[assignment]
                else:
                    raise ValueError("This should never happen")
                self._prediction_history[subject][-1].append(_prediction)
            else:
                if prediction.size()[0] > 1:
                    _prediction = tuple(float(pred) for pred in prediction.cpu().tolist())
                elif prediction.size()[0] == 1:
                    _prediction = float(prediction)  # type: ignore[assignment]
                else:
                    raise ValueError("This should never happen")
                self._prediction_history[subject] = [[_prediction]]

        # Store the corresponding subjects, if provided
        if subjects is not None:
            self._epoch_subjects.extend(subjects)

    def on_epoch_end(self, *, subjects_info=None, verbose=True, verbose_sub_groups=False,
                     verbose_variables=False) -> None:
        """
        Updates the metrics, and should be called after each epoch

        Parameters
        ----------
        subjects_info
        verbose
        verbose_sub_groups
        verbose_variables

        Returns
        -------
        None

        Examples
        --------
        >>> my_history = Histories(metrics="regression", splits={"dataset_name": ("D1", "D2", "D3")},
        ...                        expected_variables={"age": ("D1", "D3"), "ravlt": ("D2", "D3")},
        ...                        variable_metrics={"age": ("pearson_r",), "ravlt": ("spearman_rho",)})
        >>> my_y = torch.unsqueeze(torch.tensor([1.0, 3.7, 9.3, 0.2, 3.8, 4.4, 9.8, 2.3, 3.3, 2.3]), dim=-1)
        >>> my_yhat = torch.unsqueeze(torch.tensor([1.8, 5.0, 6.8, 9.6, 8.6, 4.6, 9.0, 4.2, 9.6, 1.9]), dim=-1)
        >>> my_subjects = (Subject("P1", "D1"), Subject("P3", "D3"), Subject("P2", "D1"), Subject("P4", "D2"),
        ...                Subject("P2", "D2"), Subject("P3", "D2"), Subject("P3", "D1"), Subject("P1", "D2"),
        ...                Subject("P2", "D3"), Subject("P1", "D3"))
        >>> my_info = {Subject("P1", "D1"): {"age": 37}, Subject("P2", "D1"): {"age": 34},
        ...            Subject("P3", "D1"): {"age": 15}, Subject("P1", "D2"): {"ravlt": 3},
        ...            Subject("P2", "D2"): {"ravlt": 14}, Subject("P3", "D2"): {"ravlt": 11},
        ...            Subject("P4", "D2"): {"ravlt": 15}, Subject("P1", "D3"): {"age": 36, "ravlt": 10},
        ...            Subject("P2", "D3"): {"age": 23, "ravlt": 7}, Subject("P3", "D3"): {"age": 85, "ravlt": 4}}
        >>> # Fake two batches
        >>> my_history.store_batch_evaluation(y_pred=my_yhat[:6], y_true=my_y[:6], subjects=my_subjects[:6])
        >>> my_history.store_batch_evaluation(y_pred=my_yhat[6:], y_true=my_y[6:], subjects=my_subjects[6:])
        >>> my_history.on_epoch_end(subjects_info=my_info, verbose=False, verbose_variables=False)
        >>> my_history._variables_history  # doctest: +SKIP
        """
        self._update_metrics(subjects_info=subjects_info)

        # Create an empty list for the next epoch for the prediction history
        if self._subgroup_histories is not None:
            for epoch_history in self._prediction_history.values():
                epoch_history.append([])

        # Printing
        if verbose:
            self._print_newest_metrics()
        if verbose_sub_groups and self._subgroup_histories is not None:
            self._print_newest_subgroups_metrics()
        if verbose_variables and self._variables_history is not None:
            self._print_newest_variables_metrics()

    def _print_newest_metrics(self) -> None:
        """Method for printing the newest metrics"""
        for i, (metric_name, metric_values) in enumerate(self.history.items()):
            if i == len(self.history) - 1:
                if self._name is None:
                    print(f"{metric_name}: {metric_values[-1]:.3f}")
                else:
                    print(f"{self._name}_{metric_name}: {metric_values[-1]:.3f}")
            else:
                if self._name is None:
                    print(f"{metric_name}: {metric_values[-1]:.3f}\t\t", end="")
                else:
                    print(f"{self._name}_{metric_name}: {metric_values[-1]:.3f}\t\t", end="")

    def _print_newest_subgroups_metrics(self):
        if self._subgroup_histories is not None:
            for subgroups in self._subgroup_histories.values():
                # A value in subgroups could e.g. be {"red_bull": {"mse": [val1, val2], "mae": [val3, val4]}}
                for sub_group_name, sub_group_metrics in subgroups.items():
                    for i, (metric_name, metric_values) in enumerate(sub_group_metrics.items()):
                        # With the current implementation, not all subgroups levels passed to __init__ must have been
                        # seen
                        if not metric_values:
                            continue

                        # Print metrics
                        if i == len(self.history) - 1:
                            if self._name is None:
                                print(f"{sub_group_name.lower()}_{metric_name}: {metric_values[-1]:.3f}")
                            else:
                                print(f"{self._name}_{sub_group_name.lower()}_{metric_name}: {metric_values[-1]:.3f}")
                        else:
                            if self._name is None:
                                print(f"{sub_group_name.lower()}_{metric_name}: {metric_values[-1]:.3f}\t\t", end="")
                            else:
                                print(f"{self._name}_{sub_group_name.lower()}_{metric_name}: "
                                      f"{metric_values[-1]:.3f}\t\t", end="")

    def _print_newest_variables_metrics(self):
        if self._variables_history is not None:
            for var_name, var_history in self._variables_history.items():
                for dataset_name, metrics in var_history.items():
                    for metric_name, metric_value in metrics.items():
                        if metric_name in self.get_available_groups_metrics():
                            for group, value in metric_value.items():
                                _hist_name = "" if self._name is None else f"{self._name}_"
                                print(f"{group}_{_hist_name}{var_name}_{dataset_name.lower()}_{metric_name}: "
                                      f"{value[-1]:.3f}\t\t", end="")
                        else:
                            _hist_name = "" if self._name is None else f"{self._name}_"
                            print(f"{_hist_name}{var_name}_{dataset_name.lower()}_{metric_name}: "
                                  f"{metric_value[-1]:.3f}\t\t", end="")
                print()
        if self._variables_history_ratios is not None:
            for var_name, var_history in self._variables_history_ratios.items():
                for dataset_name, metrics in var_history.items():
                    for metric_name, metric_value in metrics.items():
                        if metric_name in self.get_available_groups_metrics():
                            for group, value in metric_value.items():
                                _hist_name = "" if self._name is None else f"{self._name}_"
                                print(f"ratio_{group}_{_hist_name}{var_name}_{dataset_name.lower()}_{metric_name}: "
                                      f"{value[-1]:.3f}\t\t", end="")
                        else:
                            _hist_name = "" if self._name is None else f"{self._name}_"
                            print(f"ratio_{_hist_name}{var_name}_{dataset_name.lower()}_{metric_name}: "
                                  f"{metric_value[-1]:.3f}\t\t", end="")
                print()

    def _update_metrics(self, subjects_info):
        # Concatenate torch tenors
        y_pred = torch.cat(self._epoch_y_pred, dim=0)
        y_true = torch.cat(self._epoch_y_true, dim=0)

        # -------------
        # Update all metrics of the 'normal' history dict
        # -------------
        # We need to aggregate the predictions by averaging, and reducing the targets
        y_pred_per_subject, y_true_per_subject, subjects = _aggregate_predictions_and_ground_truths(
            subjects=self._epoch_subjects, y_true=y_true, y_pred=y_pred,
        )

        for metric, hist in self._history.items():
            hist.append(self.compute_metric(metric=metric, y_pred=y_pred_per_subject, y_true=y_true_per_subject))

        # -------------
        # (Maybe) update all metrics of all subgroups
        # -------------
        if self._subgroup_histories is not None:
            # todo: I should be able to just use what was calculated above
            # Make dictionary containing subjects combined with the prediction and target. The prediction is the average
            # of all EEG epochs/segments
            subjects_predictions: Dict[Subject, List[YYhat]] = dict()
            for subject, y_hat, y in zip(self._epoch_subjects, y_pred, y_true):
                if subject in subjects_predictions:
                    subjects_predictions[subject].append(YYhat(y_true=y, y_pred=y_hat))
                else:
                    subjects_predictions[subject] = [YYhat(y_true=y, y_pred=y_hat)]

            subjects_pred_and_true: Dict[Subject, YYhat] = dict()
            for subject, predictions_and_truths in subjects_predictions.items():
                # Verify that the ground truth is the same
                # todo: not necessarily true in self-supervised learning
                all_ground_truths = tuple(yyhat.y_true for yyhat in predictions_and_truths)
                if not all(torch.equal(all_ground_truths[0], ground_truth) for ground_truth in all_ground_truths):
                    raise ValueError("Expected all ground truths to be the same per subject, but that was not the case")

                # Set prediction to the average of all predictions, and the ground truth to the only element in the set
                _pred = torch.mean(torch.cat([torch.unsqueeze(yyhat.y_pred, dim=0)
                                              for yyhat in predictions_and_truths], dim=0), dim=0, keepdim=True)
                _true = all_ground_truths[0]
                subjects_pred_and_true[subject] = YYhat(y_pred=_pred, y_true=_true)

            # Loop through all splits
            for split_level, sub_groups in self._subgroup_histories.items():
                for sub_group_name, sub_group_metrics in sub_groups.items():
                    # A value of 'sub_group_metrics' could e.g. be {"mse": [val1, val2], "mae": [val3, val4]}

                    # Extract the subgroup
                    sub_group_subjects = tuple(subject for subject in self._epoch_subjects
                                               if subject[split_level] == sub_group_name)

                    # Exit if there are no subject in the subgroup
                    if not sub_group_subjects:
                        continue

                    # Extract their predictions and targets
                    sub_group_y_pred = torch.cat([subjects_pred_and_true[subject].y_pred
                                                  for subject in sub_group_subjects], dim=0)
                    sub_group_y_true = torch.cat([
                        torch.unsqueeze(subjects_pred_and_true[subject].y_true, dim=0)
                        if subjects_pred_and_true[subject].y_true.dim() == 0
                        else subjects_pred_and_true[subject].y_true
                        for subject in sub_group_subjects],
                        dim=0)

                    # Maybe remove redundant dimension
                    sub_group_y_pred = torch.squeeze(sub_group_y_pred, dim=-1)

                    # Loop through and calculate the metrics
                    for metric_name, metric_values in sub_group_metrics.items():
                        # Compute metrics for the subgroup and store it
                        metric_values.append(self.compute_metric(metric=metric_name, y_pred=sub_group_y_pred,
                                                                 y_true=sub_group_y_true))

        # -------------
        # Build the dicts and lists for the different features to correlate with the delta
        # -------------
        if subjects_info is not None and self._variables_history is not None:
            data_matrices, data_matrices_ratio = self._build_error_and_info_objects(
                y_pred_per_subject=y_pred_per_subject, y_true_per_subject=y_true_per_subject, subjects=subjects,
                subjects_info=subjects_info
            )
            # todo: such a quick fix, maybe this can be improved
            for var_name, epoch_history in data_matrices.items():
                for dataset_name, tensor_list in epoch_history.items():
                    for metric in self._variable_metrics[var_name]:  # type: ignore[index]
                        # Compute metric
                        _tensor = torch.cat(tensor_list, dim=0)

                        if metric in self.get_available_groups_metrics():
                            metric_value = self._compute_groups_metric(
                                metric=metric, variable=_tensor[:, 0], groups=_tensor[:, 1]
                            )

                            # Might be the first time it is seen, in that case need to initialise the lists per group
                            if not self._variables_history[var_name][dataset_name][metric]:
                                for group in metric_value:
                                    self._variables_history[var_name][dataset_name][metric][group] = []

                            # Keys should be the same for all epochs
                            for group, value in metric_value.items():
                                self._variables_history[var_name][dataset_name][metric][group].append(value)

                        else:
                            metric_value = self.compute_metric(
                                metric=metric, y_pred=_tensor[:, 0], y_true=_tensor[:, 1]
                            )

                            # Add results
                            self._variables_history[var_name][dataset_name][metric].append(metric_value)

            for var_name, epoch_history in data_matrices_ratio.items():
                for dataset_name, tensor_list in epoch_history.items():
                    for metric in self._variable_metrics[var_name]:  # type: ignore[index]
                        # Compute metric
                        _tensor = torch.cat(tensor_list, dim=0)

                        if metric in self.get_available_groups_metrics():
                            metric_value = self._compute_groups_metric(
                                metric=metric, variable=_tensor[:, 0], groups=_tensor[:, 1]
                            )

                            # Might be the first time it is seen, in that case need to initialise the lists per group
                            if not self._variables_history_ratios[var_name][dataset_name][metric]:
                                for group in metric_value:
                                    self._variables_history_ratios[var_name][dataset_name][metric][group] = []

                            # Keys should be the same for all epochs
                            for group, value in metric_value.items():
                                self._variables_history_ratios[var_name][dataset_name][metric][group].append(value)

                        else:
                            metric_value = self.compute_metric(
                                metric=metric, y_pred=_tensor[:, 0], y_true=_tensor[:, 1]
                            )

                            # Add results
                            self._variables_history_ratios[var_name][dataset_name][metric].append(metric_value)

        # -------------
        # Remove the epoch histories
        # -------------
        self._epoch_y_pred = []
        self._epoch_y_true = []
        self._epoch_subjects = []

    def _build_error_and_info_objects(self, *, y_pred_per_subject, y_true_per_subject, subjects, subjects_info):
        """
        Method for extracting delta values and info of which to (e.g.) correlate with, in a dict which makes it more
        feasible for computing these metrics

        Parameters
        ----------
        y_pred_per_subject : torch.Tensor
        y_true_per_subject : torch.Tensor
        subjects : tuple[Subject, ...]
        subjects_info : dict[Subject, dict[str, typing.Any]]

        Returns
        -------
        tuple[dict[str, dict[str, list[torch.Tensor]]], dict[str, dict[str, list[torch.Tensor]]]]

        Examples
        --------
        >>> my_history = Histories(metrics="regression", splits={"dataset_name": ("D1", "D2", "D3")},
        ...                        expected_variables={"age": ("D1", "D3"), "ravlt": ("D2", "D3")},
        ...                        variable_metrics={"age": ("pearson_r",), "ravlt": ("spearman_rho",)})
        >>> my_y = torch.tensor([1.0, 3.7, 9.3, 0.2, 3.8, 4.4, 9.8, 2.3, 3.3, 2.3])
        >>> my_yhat = torch.tensor([1.8, 5.0, 6.8, 9.6, 8.6, 4.6, 9.0, 4.2, 9.6, 1.9])
        >>> my_subjects = (Subject("P1", "D1"), Subject("P3", "D3"), Subject("P2", "D1"), Subject("P4", "D2"),
        ...                Subject("P2", "D2"), Subject("P3", "D2"), Subject("P3", "D1"), Subject("P1", "D2"),
        ...                Subject("P2", "D3"), Subject("P1", "D3"))
        >>> my_info = {Subject("P1", "D1"): {"age": 37}, Subject("P2", "D1"): {"age": 34},
        ...            Subject("P3", "D1"): {"age": 15}, Subject("P1", "D2"): {"ravlt": 3},
        ...            Subject("P2", "D2"): {"ravlt": 14}, Subject("P3", "D2"): {"ravlt": 11},
        ...            Subject("P4", "D2"): {"ravlt": 15}, Subject("P1", "D3"): {"age": 36, "ravlt": 10},
        ...            Subject("P2", "D3"): {"age": 23, "ravlt": 7}, Subject("P3", "D3"): {"age": 85, "ravlt": 4}}
        >>> my_history._build_error_and_info_objects(
        ...     y_pred_per_subject=my_yhat, y_true_per_subject=my_y, subjects=my_subjects, subjects_info=my_info
        ... )  # doctest: +NORMALIZE_WHITESPACE
        ({'age': {'All': [tensor([[-0.8000, 37.0000]]), tensor([[-1.3000, 85.0000]]), tensor([[ 2.5000, 34.0000]]),
                          tensor([[ 0.8000, 15.0000]]), tensor([[-6.3000, 23.0000]]), tensor([[ 0.4000, 36.0000]])],
                  'D1': [tensor([[-0.8000, 37.0000]]), tensor([[ 2.5000, 34.0000]]), tensor([[ 0.8000, 15.0000]])],
                  'D3': [tensor([[-1.3000, 85.0000]]), tensor([[-6.3000, 23.0000]]), tensor([[ 0.4000, 36.0000]])]},
          'ravlt': {'All': [tensor([[-1.3000,  4.0000]]), tensor([[-9.4000, 15.0000]]), tensor([[-4.8000, 14.0000]]),
                            tensor([[-0.2000, 11.0000]]), tensor([[-1.9000,  3.0000]]), tensor([[-6.3000,  7.0000]]),
                            tensor([[ 0.4000, 10.0000]])],
                    'D2': [tensor([[-9.4000, 15.0000]]), tensor([[-4.8000, 14.0000]]), tensor([[-0.2000, 11.0000]]),
                            tensor([[-1.9000,  3.0000]])],
                    'D3': [tensor([[-1.3000,  4.0000]]), tensor([[-6.3000,  7.0000]]), tensor([[ 0.4000, 10.0000]])]}},
         {'age': {'All': [tensor([[ 0.5556, 37.0000]]), tensor([[ 0.7400, 85.0000]]), tensor([[ 1.3676, 34.0000]]),
                          tensor([[ 1.0889, 15.0000]]), tensor([[ 0.3437, 23.0000]]), tensor([[ 1.2105, 36.0000]])],
                  'D1': [tensor([[ 0.5556, 37.0000]]), tensor([[ 1.3676, 34.0000]]), tensor([[ 1.0889, 15.0000]])],
                  'D3': [tensor([[ 0.7400, 85.0000]]), tensor([[ 0.3437, 23.0000]]), tensor([[ 1.2105, 36.0000]])]},
          'ravlt': {'All': [tensor([[0.7400, 4.0000]]), tensor([[ 0.0208, 15.0000]]), tensor([[ 0.4419, 14.0000]]),
                            tensor([[ 0.9565, 11.0000]]), tensor([[0.5476, 3.0000]]), tensor([[0.3437, 7.0000]]),
                            tensor([[ 1.2105, 10.0000]])],
                    'D2': [tensor([[ 0.0208, 15.0000]]), tensor([[ 0.4419, 14.0000]]), tensor([[ 0.9565, 11.0000]]),
                           tensor([[0.5476, 3.0000]])], 'D3': [tensor([[0.7400, 4.0000]]), tensor([[0.3437, 7.0000]]),
                           tensor([[ 1.2105, 10.0000]])]}})
        """
        if self._variables_history is None:
            raise RuntimeError("Cannot collate prediction errors and variables when no variables are expected")

        # I have to stay compatible with Python 3.8, so I'll do a length check instead of strict=True in zip. Could
        # probably remove the check though...
        if not (y_pred_per_subject.size()[0] == y_true_per_subject.size()[0] == len(subjects)):
            raise ValueError(f"Expected number of predictions, number of ground truths, and number of subjects, to be "
                             f"the same, but found {y_pred_per_subject.size()[0]}, {y_true_per_subject.size()[0]}, and "
                             f"{len(subjects)}")
        # E.g. {"age": {"All": [tensor(0.4, 45), tensor(0.6, 34)], "SRM": [tensor(0.4, 45)], "Lemon": [tensor(0.6, 34)},
        #       "ravlt: {"All": [tensor(0.4, 14)], "SRM": [tensor(0.4, 14)]}}
        # That is, the first element of the tensor is the delta y, the second element is the variable value
        data_matrices: Dict[str, Dict[str, List[torch.Tensor]]] = {
            var_name: {dataset_name: [] for dataset_name in dataset_histories}
            for var_name, dataset_histories in self._variables_history.items()
        }

        # Now, I'll also add y_hat / t_true. That is, the first element of the tensor is the ration, the second element
        # is the variable value
        data_matrices_ratio: Dict[str, Dict[str, List[torch.Tensor]]] = {
            var_name: {dataset_name: [] for dataset_name in dataset_histories}
            for var_name, dataset_histories in self._variables_history.items()
        }
        for delta_y, ratio, subject in zip(y_true_per_subject - y_pred_per_subject,
                                           y_true_per_subject / y_pred_per_subject,
                                           subjects):
            for var_name, dataset_names in self._variables_history.items():
                if subject.dataset_name in dataset_names:
                    # Add to the dataset
                    data_matrices[var_name][subject.dataset_name].append(
                        torch.tensor([[delta_y, subjects_info[subject][var_name]]])
                    )
                    data_matrices_ratio[var_name][subject.dataset_name].append(
                        torch.tensor([[ratio, subjects_info[subject][var_name]]])
                    )

                    # Add to the "All" dataset as well
                    data_matrices[var_name]["All"].append(
                        torch.tensor([[delta_y, subjects_info[subject][var_name]]])
                    )
                    # Add to the "All" dataset as well
                    data_matrices_ratio[var_name]["All"].append(
                        torch.tensor([[ratio, subjects_info[subject][var_name]]])
                    )

        return data_matrices, data_matrices_ratio

    @classmethod
    def _compute_metric(cls, metric: str, *, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Method for computing the specified metric"""

        # Input check
        if metric not in cls.get_available_metrics():
            raise ValueError(f"The metric {metric} was not recognised. The available ones are: "
                             f"{cls.get_available_metrics()}")

        # Compute the metric
        return getattr(cls, metric)(y_pred=y_pred, y_true=y_true)

    @classmethod
    def compute_metric(cls, metric, *, y_pred, y_true):
        """
        Computes a specified evaluation metric.

        Notes
        -----
        - NaN values in the prediction MAY occur due to numerical instabilities when training DL models, see
          https://stackoverflow.com/questions/33962226/common-causes-of-nans-during-training-of-neural-networks. If NaNs
          are detected, some scores raise an error, others return nan.
        - Since y_pred.isnan().any() did not always work, did not consistently detect NaN values, a `ValueError` is
          caught instead. This (unfortunately) means that NaN values in `y_pred` or `y_true` will trigger the same
          custom error, if a ValueError was originally raised in the sklearn implementation

        Parameters
        ----------
        metric : str
        y_pred : torch.Tensor
        y_true : torch.Tensor

        Returns
        -------
        float
        """
        # Ensure torch tensor
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true)

        # Compute score
        try:
            return cls._compute_metric(metric=metric, y_pred=y_pred, y_true=y_true)
        except ValueError as e:
            if "NaN" in str(e):  # Can't make it too specific due to minor differences in versions
                raise NaNValueError
            raise e

    @classmethod
    def _compute_groups_metric(cls, metric: str, *, variable: torch.Tensor, groups: torch.Tensor):
        """Method for computing the specified groups metric"""

        # Input check
        if metric not in cls.get_available_groups_metrics():
            raise ValueError(f"The metric {metric} was not recognised. The available ones are: "
                             f"{cls.get_available_groups_metrics()}")

        # Compute the metric
        return getattr(cls, metric)(variable=variable, groups=groups)

    # -----------------
    # Properties
    # -----------------
    @property
    def name(self) -> str:
        # todo: I don't like this, looks like a quick fix...
        return "UNNAMED" if self._name is None else self._name  # type: ignore[no-any-return]

    @property
    def history(self):
        # Returning values as tuples to make immutability
        return {metric_name: tuple(performance_scores) for metric_name, performance_scores in self._history.items()}

    @property
    def newest_metrics(self):
        return {metric_name: performance[-1] for metric_name, performance in self._history.items()}

    @property
    def has_variables_history(self) -> bool:
        return self._variables_history is not None

    # -----------------
    # Methods for saving
    # -----------------
    def save_prediction_history(self, history_name, path, decimals=3):
        """
        Method for saving the predictions in a csv file

        Parameters
        ----------
        history_name : str
        path : str
        decimals : int
            Number of decimal places for storing the predictions

        Returns
        -------
        None
        """
        # --------------
        # Remove the last 'epoch' if it is empty
        # --------------
        for epochs in self._prediction_history.values():
            if not epochs[-1]:
                del epochs[-1]

        # --------------
        # Sanity checks
        # --------------
        # Check if number of epochs is the same for all subjects
        all_num_epochs = set(len(epoch_history) for epoch_history in self._prediction_history.values())

        assert len(all_num_epochs) == 1, (f"Expected number of DL epochs to be the same for all epochs and subjects, "
                                          f"but found {all_num_epochs}")
        num_epochs = tuple(all_num_epochs)[0]

        # Check if number of EEG epochs is the same for all subjects and epochs
        all_num_eeg_epochs = set(len(eeg_epoch_predictions) for epoch_predictions in self._prediction_history.values()
                                 for eeg_epoch_predictions in epoch_predictions)
        assert len(all_num_eeg_epochs) == 1, (f"Expected number of EEG epochs to be the same for all epochs and "
                                              f"subjects, but found {all_num_eeg_epochs}")
        num_eeg_epochs = tuple(all_num_eeg_epochs)[0]

        # --------------
        # Saving of prediction history
        # --------------
        # Flatten out the prediction history
        prediction_history = {sub_id: tuple(itertools.chain(*epoch_predictions))
                              for sub_id, epoch_predictions in self._prediction_history.items()}

        # If the predictions of the model is a vector (such as a domain discriminator), further flattening is needed
        if all(isinstance(prediction, tuple) for predictions in prediction_history.values()
               for prediction in predictions):
            output_dims = set(len(prediction) for predictions in prediction_history.values()  # type: ignore[arg-type]
                              for prediction in predictions)
            assert len(output_dims) == 1, (f"Expected output dimensions to be the same for all predictions, but found "
                                           f"{output_dims}")
            output_dim = tuple(output_dims)[0]

            prediction_history = {sub_id: tuple(itertools.chain(*predictions))  # type: ignore[arg-type]
                                  for sub_id, predictions in prediction_history.items()}
            epochs_column_names = [f"dim{k}_pred{j + 1}_epoch{i + 1}" for i in range(num_epochs) for j in
                                   range(num_eeg_epochs) for k in range(output_dim)]
        else:
            epochs_column_names = [f"pred{j+1}_epoch{i+1}" for i in range(num_epochs) for j in range(num_eeg_epochs)]

        # Create pandas dataframe with the prediction histories
        df = pandas.DataFrame.from_dict(prediction_history, orient="index", columns=epochs_column_names)

        # Add dataset and subject ID
        df.insert(loc=0, value=tuple(subject.subject_id for subject in self._prediction_history),  # type: ignore
                  column="sub_id")
        df.insert(loc=0, value=tuple(subject.dataset_name for subject in self._prediction_history),  # type: ignore
                  column="dataset")

        # Drop the index
        df.reset_index(inplace=True, drop=True)

        # Round the predictions
        df = df.round({col: decimals for col in epochs_column_names})

        # Save csv file and make it read-only
        to_path = os.path.join(path, f"{history_name}_predictions.csv")
        df.to_csv(to_path, index=False)
        os.chmod(to_path, 0o444)

    def save_subgroup_metrics(self, history_name, path, *, save_plots, decimals, fig_size=(12, 6), font_size=15,
                              title_fontsize=20):
        # If there are no subgroups registered, raise a warning and do nothing
        if self._subgroup_histories is None:
            warnings.warn(message="Tried to save plot of metrics computed per sub-group, but there were no subgroups",
                          category=PlotNotSavedWarning)
            return

        # Loop through all levels
        for level, subgroups in self._subgroup_histories.items():
            # Get the metrics
            metrics: Tuple[str, ...] = ()
            for subgroup_metrics in subgroups.values():
                if not metrics:
                    metrics = tuple(subgroup_metrics.keys())
                else:
                    # All metrics used should be the same, but different order is ok
                    if not set(metrics) == set(subgroup_metrics.keys()):
                        raise RuntimeError("Expected all metrics to be the same for all sub-groups, but that was not "
                                           "the case")

            # Create folder
            level_path = os.path.join(path, level)
            if not os.path.isdir(level_path):
                os.mkdir(level_path)

            # Loop through and create a plot per metrics (and level)
            for metric_to_plot in metrics:
                # Make folder
                metric_path = os.path.join(level_path, metric_to_plot)
                if not os.path.isdir(metric_path):
                    os.mkdir(metric_path)

                # Loop through all subgroups
                df_dict: Dict[str, List[float]] = dict()
                for subgroup_name, subgroup_metrics in subgroups.items():
                    # Get the performance
                    performance = subgroup_metrics[metric_to_plot]

                    # Only add it if it is non-empty
                    if performance:
                        df_dict[subgroup_name] = performance

                    # Plot, if values are registered
                    if performance:
                        pyplot.plot(range(1, len(performance) + 1), performance, label=subgroup_name)

                # Plotting
                if save_plots:
                    pyplot.figure(figsize=fig_size)

                    pyplot.title(f"Performance (level={level})", fontsize=title_fontsize)
                    pyplot.xlabel("Epoch", fontsize=font_size)
                    pyplot.ylabel(metric_to_plot.capitalize(), fontsize=font_size)
                    pyplot.tick_params(labelsize=font_size)
                    pyplot.legend(fontsize=font_size)
                    pyplot.grid()

                    # Save figure and close it
                    pyplot.savefig(os.path.join(metric_path, f"{history_name}_{metric_to_plot}.png"))
                    pyplot.close()

                # Save history object as well
                df = pandas.DataFrame.from_dict(df_dict)

                if decimals is not None:
                    df = df.round(decimals)

                to_path = os.path.join(metric_path, f"{history_name}_{metric_to_plot}.csv")
                df.to_csv(to_path, index=False)
                os.chmod(to_path, 0o444)  # Make read-only

    def save_main_history(self, history_name, path, decimals):
        """Method for saving the main (non subgroup) history"""
        # ---------------
        # Save predictions
        # ---------------
        self.save_prediction_history(history_name=history_name, path=path, decimals=decimals)

        # ---------------
        # Save the metrics in .csv format
        # ---------------
        # Create pandas dataframe
        df = pandas.DataFrame(self._history)

        # Maybe set decimals
        if decimals is not None:
            df = df.round(decimals)

        # Save as .csv
        to_path = os.path.join(path, f"{history_name}_metrics.csv")
        df.to_csv(to_path, index=False)

        # Make read-only
        os.chmod(to_path, 0o444)

    def save_variables_histories(self, history_name, path, decimals, save_plots):
        # If there is no history, raise a warning and do nothing
        if self._variables_history is None:
            warnings.warn("Tried to save results of associations with prediction error and other variables, but there "
                          "were no such history", PlotNotSavedWarning)
            return

        # Associations with difference between predicted and true value
        error_difference_path = path / "difference"
        if not os.path.isdir(error_difference_path):
            os.mkdir(error_difference_path)
        self._save_variables_history(
            history=self._variables_history, history_name=history_name, path=error_difference_path, decimals=decimals,
            save_plots=save_plots
        )

        # Associations with ratio between predicted and true value
        error_ratio_path = path / "ratio"
        if not os.path.isdir(error_ratio_path):
            os.mkdir(error_ratio_path)
        self._save_variables_history(
            history=self._variables_history_ratios, history_name=history_name, path=error_ratio_path, decimals=decimals,
            save_plots=save_plots
        )

    def _save_variables_history(self, history, history_name, path, decimals, save_plots, *, fig_size=(12, 6),
                                font_size=15, title_fontsize=20):
        for var_name, var_history in history.items():
            # I'll have a new folder for every variable (e.g., age, ravlt_tot, etc.)
            var_path = path / var_name
            if not os.path.isdir(var_path):
                os.mkdir(path / var_name)

            # Convert to a dict which is feasible to make a Dataframe of
            metrics_dict: Dict[str, Dict[str, List[float]]] = {}  # {"pearson_r": {"All": [...], "SRM": [...]}}
            for dataset_name, metrics_history in var_history.items():
                for metric, history_list in metrics_history.items():
                    if metric in self.get_available_groups_metrics():
                        for group, hist_list in history_list.items():
                            _metric = f"{metric}_{group}"
                            if _metric not in metrics_dict:
                                metrics_dict[_metric] = {}

                            metrics_dict[_metric][dataset_name] = hist_list
                    else:
                        if metric not in metrics_dict:
                            metrics_dict[metric] = {}

                        metrics_dict[metric][dataset_name] = history_list

            # Loop through our newly created dictionary to make plots and save DataFrames
            for metric, dataset_histories in metrics_dict.items():
                if save_plots:
                    pyplot.figure(figsize=fig_size)

                    for dataset_name, history_list in dataset_histories.items():
                        # Plot
                        pyplot.plot(range(1, len(history_list) + 1), history_list, label=dataset_name)

                        # Plot cosmetics
                        pyplot.title(f"Associations with prediction error: {var_name}", fontsize=title_fontsize)
                        pyplot.xlabel("Epoch", fontsize=font_size)
                        pyplot.ylabel(metric.capitalize(), fontsize=font_size)
                        pyplot.tick_params(labelsize=font_size)
                        pyplot.legend(fontsize=font_size)
                        pyplot.grid()

                    # Save figure and close it
                    pyplot.savefig(var_path / f"{history_name}_{metric}.png")
                    pyplot.close()

                # Make and save dataframe
                df = pandas.DataFrame.from_dict(dataset_histories)

                df = df.round(decimals)

                df.to_csv(var_path / f"{history_name}_{metric}.csv", index=False)

    # -----------------
    # Methods for getting the available metrics
    # -----------------
    @classmethod
    def get_available_regression_metrics(cls):
        """Get all regression metrics available for the class. The regression metric must be a method
        decorated by @regression_metric to be properly registered"""
        # Get all regression metrics
        metrics: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a regression metric
            if callable(attribute) and getattr(attribute, "_is_regression_metric", False):
                metrics.append(method)

        # Convert to tuple and return
        return tuple(metrics)

    @classmethod
    def get_available_classification_metrics(cls):
        """Get all classification metrics available for the class. The classification metric must be a method
        decorated by @classification_metric to be properly registered"""
        # Get all classification metrics
        metrics: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a classification metric
            if callable(attribute) and getattr(attribute, "_is_classification_metric", False):
                metrics.append(method)

        # Convert to tuple and return
        return tuple(metrics)

    @classmethod
    def get_available_multiclass_classification_metrics(cls):
        """Get all multiclass classification metrics available for the class. The classification metric must be a method
        decorated by @multiclass_classification_metric to be properly registered"""
        # Get all classification metrics
        metrics: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a classification metric
            if callable(attribute) and getattr(attribute, "_is_multiclass_classification_metric", False):
                metrics.append(method)

        # Convert to tuple and return
        return tuple(metrics)

    @classmethod
    def get_available_groups_metrics(cls):
        """Get all groups metrics available for the class. The metrics must be a method decorated by @groups_metric to
        be properly registered"""
        # Get all groups metrics
        metrics: List[str] = []
        for method in dir(cls):
            attribute = getattr(cls, method)

            # Append (as type 'str') if it is a groups metric
            if callable(attribute) and getattr(attribute, "_is_groups_metric", False):
                metrics.append(method)

        # Convert to tuple and return
        return tuple(metrics)

    @classmethod
    def get_available_metrics(cls):
        """Get all available metrics. Groups metrics are not added as they are not performance metrics per sé"""
        return (cls.get_available_classification_metrics() + cls.get_available_multiclass_classification_metrics() +
                cls.get_available_regression_metrics())

    @classmethod
    def get_default_metrics(cls, metrics: str):
        """Method for getting pre-specified sets of metrics"""
        if metrics == "regression":
            return cls.get_available_regression_metrics()
        elif metrics == "classification":
            return cls.get_available_classification_metrics()
        elif metrics == "multiclass_classification":
            return cls.get_available_multiclass_classification_metrics()
        else:
            raise ValueError(f"The requested pre-specified set of metrics was not understood: {metrics}")

    # -----------------
    # Regression metrics
    # -----------------
    @staticmethod
    @regression_metric(is_higher_better=False)
    def mse(y_pred: torch.Tensor, y_true: torch.Tensor):
        return mean_squared_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    @staticmethod
    @regression_metric(is_higher_better=False)
    def mae(y_pred: torch.Tensor, y_true: torch.Tensor):
        return mean_absolute_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    @staticmethod
    @regression_metric(is_higher_better=False)
    def med_ae(y_pred: torch.Tensor, y_true: torch.Tensor):
        return median_absolute_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    @staticmethod
    @regression_metric(is_higher_better=False)
    def max_error(y_pred: torch.Tensor, y_true: torch.Tensor):
        return max_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    @staticmethod
    @regression_metric(is_higher_better=False)
    def mape(y_pred: torch.Tensor, y_true: torch.Tensor):
        return mean_absolute_percentage_error(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    @staticmethod
    @regression_metric(is_higher_better=True)
    def pearson_r(y_pred: torch.Tensor, y_true: torch.Tensor):
        # Removing redundant dimension may be necessary
        if y_true.dim() == 2:
            y_true = torch.squeeze(y_true, dim=1)
        if y_pred.dim() == 2:
            y_pred = torch.squeeze(y_pred, dim=1)

        # Compute and return
        return pearsonr(x=y_true.cpu(), y=y_pred.cpu())[0]

    @staticmethod
    @regression_metric(is_higher_better=True)
    def spearman_rho(y_pred: torch.Tensor, y_true: torch.Tensor):
        # Removing redundant dimension may be necessary
        if y_true.dim() == 2:
            y_true = torch.squeeze(y_true, dim=1)
        if y_pred.dim() == 2:
            y_pred = torch.squeeze(y_pred, dim=1)

        # Compute and return
        # noinspection PyTypeChecker
        return spearmanr(a=y_true.cpu(), b=y_pred.cpu())[0]

    @staticmethod
    @regression_metric(is_higher_better=True)
    def conc_cc(y_pred: torch.Tensor, y_true: torch.Tensor):
        """Concordance correlation coefficient (https://en.wikipedia.org/wiki/Concordance_correlation_coefficient)"""
        with torch.no_grad():
            # Removing redundant dimension may be necessary
            if y_true.dim() == 2:
                y_true = torch.squeeze(y_true, dim=1)
            if y_pred.dim() == 2:
                y_pred = torch.squeeze(y_pred, dim=1)

            # Make some computations
            mean_true = torch.mean(y_true)
            mean_pred = torch.mean(y_pred)

            var_true = torch.var(y_true, unbiased=False)  # Using population variance
            var_pred = torch.var(y_pred, unbiased=False)

            cov = torch.mean((y_true - mean_true) * (y_pred - mean_pred))  # Covariance

            numerator = 2 * cov
            denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
            score = numerator / denominator

        return score.item()

    @staticmethod
    @regression_metric(is_higher_better=True)
    def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor):
        return r2_score(y_true=y_true.cpu(), y_pred=y_pred.cpu())

    @staticmethod
    @regression_metric(is_higher_better=True)
    def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor):
        """Similar to r2_score, but does not account for systematic offset in the prediction (see sklearn
        documentation)"""
        return explained_variance_score(y_true=y_true.cpu(), y_pred=y_pred.cpu(), force_finite=True)

    # -----------------
    # Classification metrics
    # -----------------
    @staticmethod
    @classification_metric(is_higher_better=True)
    def auc(y_pred: torch.Tensor, y_true: torch.Tensor):
        # todo: a value error is raised if only one class is present in y_true
        try:
            return roc_auc_score(y_true=torch.squeeze(y_true, dim=-1).cpu(),
                                 y_score=torch.squeeze(y_pred, dim=-1).cpu())
        except ValueError:
            return numpy.nan

    # -----------------
    # Multiclass classification metrics
    #
    # Note that we assume the logits, not the actual probabilities from softmax
    # -----------------
    @staticmethod
    @multiclass_classification_metric(is_higher_better=True)
    def acc(y_pred: torch.Tensor, y_true: torch.Tensor):
        return accuracy_score(y_pred=y_pred.cpu().argmax(dim=-1), y_true=y_true.cpu())

    @staticmethod
    @multiclass_classification_metric(is_higher_better=True)
    def balanced_acc(y_pred: torch.Tensor, y_true: torch.Tensor):
        return balanced_accuracy_score(y_pred=y_pred.cpu().argmax(dim=-1), y_true=y_true.cpu())

    @staticmethod
    @multiclass_classification_metric(is_higher_better=True)
    def mcc(y_pred: torch.Tensor, y_true: torch.Tensor):
        return matthews_corrcoef(y_pred=y_pred.cpu().argmax(dim=-1), y_true=y_true.cpu())

    @staticmethod
    @multiclass_classification_metric(is_higher_better=True)
    def kappa(y_pred: torch.Tensor, y_true: torch.Tensor):
        return cohen_kappa_score(y1=y_pred.cpu().argmax(dim=-1), y2=y_true.cpu())

    @staticmethod
    @multiclass_classification_metric(is_higher_better=True)
    def auc_ovo(y_pred: torch.Tensor, y_true: torch.Tensor):
        try:
            with torch.no_grad():
                return roc_auc_score(y_true=y_true.cpu(), y_score=torch.softmax(y_pred, dim=-1).cpu(),
                                     multi_class="ovo")
        except ValueError as e:
            if "number of classes" in str(e).lower():
                # Raise if number of classes in y_true not equal to the number of columns in 'y_score'
                raise MismatchClassCountError
            raise e

    @staticmethod
    @multiclass_classification_metric(is_higher_better=True)
    def auc_ovr(y_pred: torch.Tensor, y_true: torch.Tensor):
        try:
            with torch.no_grad():
                return roc_auc_score(y_true=y_true.cpu(), y_score=torch.softmax(y_pred, dim=-1).cpu(),
                                     multi_class="ovr")
        except ValueError as e:
            if "number of classes" in str(e).lower():
                # Raise if number of classes in y_true not equal to the number of columns in 'y_score'
                raise MismatchClassCountError
            raise e

    @staticmethod
    @multiclass_classification_metric(is_higher_better=False)
    def ce_loss(y_pred: torch.Tensor, y_true: torch.Tensor):
        with torch.no_grad():
            # y_true must have dtype = long or int64
            performance = nn.CrossEntropyLoss(reduction='mean')(y_pred, y_true).cpu()
        return float(performance)

    # -----------------
    # Groups metrics
    #
    # These are designed to compute statistics on groups, such as computing the mean y / y_hat for each clinical
    # status. Could probably benefit from a refactoring and re-structuring of the code, but as per now I don't want to
    # -----------------
    @staticmethod
    @groups_metric
    def std(variable: torch.Tensor, groups: torch.Tensor):
        # Ensure 1D
        if variable.dim() == 2:
            variable = torch.squeeze(variable, dim=1)
        if groups.dim() == 2:
            groups = torch.squeeze(groups, dim=1)

        unique_groups = torch.unique(groups)
        return {str(int(group)): torch.std(variable[groups == group]).item() for group in unique_groups}

    @staticmethod
    @groups_metric
    def mean(variable: torch.Tensor, groups: torch.Tensor):
        """
        Computes the mean of a variable per group

        Parameters
        ----------
        variable : torch.Tensor
        groups : torch.Tensor

        Returns
        -------
        dict[str, float]

        Examples
        --------
        >>> my_var = torch.tensor([0.81, 0.90, 0.12, 0.91, 0.63, 0.09, 0.27, 0.54, 0.95, 0.96])
        >>> my_groups = torch.tensor([0, 1, 2, 2, 2, 1, 1, 0, 2, 2])
        >>> Histories.mean(my_var, my_groups)  # doctest: +ELLIPSIS
        {'0': 0.6750..., '1': 0.41999..., '2': 0.71399...}
        """
        # Ensure 1D
        if variable.dim() == 2:
            variable = torch.squeeze(variable, dim=1)
        if groups.dim() == 2:
            groups = torch.squeeze(groups, dim=1)

        unique_groups = torch.unique(groups)
        return {str(int(group)): torch.mean(variable[groups == group]).item() for group in unique_groups}

    @classmethod
    @groups_metric
    def pairwise_mean_difference(cls, variable: torch.Tensor, groups: torch.Tensor):
        """
        Computes the pairwise difference between means of groups

        Parameters
        ----------
        variable : torch.Tensor
        groups : torch.Tensor

        Returns
        -------
        dict[str, float]

        Examples
        --------
        >>> my_var = torch.tensor([0.81, 0.12, 0.90, 0.91, 0.63, 0.09, 0.27, 0.54, 0.95, 0.96])
        >>> my_groups = torch.tensor([0, 2, 1, 2, 2, 1, 1, 0, 2, 2])
        >>> Histories.pairwise_mean_difference(my_var, my_groups)  # doctest: +ELLIPSIS
        {'0-1': 0.2550..., '0-2': -0.03899..., '1-2': -0.29399...}
        """
        # Ensure 1D
        if variable.dim() == 2:
            variable = torch.squeeze(variable, dim=1)
        if groups.dim() == 2:
            groups = torch.squeeze(groups, dim=1)

        # Compute group means
        group_means = cls.mean(variable=variable, groups=groups)

        # Compute pairwise differences
        pairwise_mean_difference: Dict[str, float] = {}
        for (group_1, mean_1), (group_2, mean_2) in itertools.combinations(group_means.items(), 2):
            # torch.unique returns sorted tensors, so this should be fine
            pairwise_mean_difference[f"{group_1}-{group_2}"] = mean_1 - mean_2

        return pairwise_mean_difference

    @staticmethod
    @groups_metric
    def median(variable: torch.Tensor, groups: torch.Tensor):
        # Ensure 1D
        if variable.dim() == 2:
            variable = torch.squeeze(variable, dim=1)
        if groups.dim() == 2:
            groups = torch.squeeze(groups, dim=1)

        unique_groups = torch.unique(groups)
        return {str(int(group)): torch.median(variable[groups == group]).item() for group in unique_groups}

    @classmethod
    @groups_metric
    def pairwise_median_difference(cls, variable: torch.Tensor, groups: torch.Tensor):
        # Ensure 1D
        if variable.dim() == 2:
            variable = torch.squeeze(variable, dim=1)
        if groups.dim() == 2:
            groups = torch.squeeze(groups, dim=1)

        # Compute group medians
        group_medians = cls.median(variable=variable, groups=groups)

        # Compute pairwise differences
        pairwise_median_difference: Dict[str, float] = {}
        for (group_1, median_1), (group_2, median_2) in itertools.combinations(group_medians.items(), 2):
            # torch.unique returns sorted tensors, so this should be fine
            pairwise_median_difference[f"{group_1}-{group_2}"] = median_1 - median_2

        return pairwise_median_difference


# ----------------
# Warnings and exceptions
# ----------------
class PlotNotSavedWarning(UserWarning):
    ...


class NaNValueError(Exception):
    """Should be raised when the predictions of a model contain NaN values."""


class MismatchClassCountError(Exception):
    """Should be raised instead of ValueError, as is done in sklearn, for multiclass classification metrics when number
    of classes in y_pred and y_true does not match"""


# ----------------
# Functions
# ----------------
def higher_is_better(metric):
    """
    Function for determining if a metric is 'higher the better' or 'lower the better'. If True, 'higher the better'
    applies, if False, 'lower the better', if the metric is not recognised, an error is raised

    Parameters
    ----------
    metric : str

    Returns
    -------
    bool

    Examples
    --------
    >>> higher_is_better("r2_score")
    True
    >>> higher_is_better("ce_loss")
    False
    >>> higher_is_better("med_ae")
    False
    >>> higher_is_better("balanced_acc")
    True
    >>> higher_is_better("auc_ovr")
    True
    >>> higher_is_better("conc_cc")
    True
    """
    # Try to get the metric function using the name
    metric_func = getattr(Histories, metric, None)

    # Check if the function exists and has the '_is_higher_better' attribute
    if metric_func is None:
        raise ValueError(f"Expected the metric to be in {Histories.get_available_metrics()}, but found '{metric}'")

    if hasattr(metric_func, "_higher_is_better"):
        return getattr(metric_func, "_higher_is_better")
    else:
        raise ValueError(f"Metric {metric!r} has not been properly decorated, as we can't infer its interpretation "
                         f"(higher or lower is better).")


def _aggregate_predictions_and_ground_truths(*, subjects, y_pred, y_true):
    """Function for aggregating predictions when predictions have been made for multiple EEG epochs per subject. This
    function computes the new prediction as the average of all, and also checks that the ground truth is always the same
    per subject"""
    # Make dictionary containing subjects combined with the prediction and target
    subjects_predictions: Dict[Subject, List[YYhat]] = dict()
    for subject, y_hat, y in zip(subjects, y_pred, y_true):
        if subject in subjects_predictions:
            subjects_predictions[subject].append(YYhat(y_true=y, y_pred=y_hat))
        else:
            subjects_predictions[subject] = [YYhat(y_true=y, y_pred=y_hat)]

    subjects_pred_and_true: Dict[Subject, YYhat] = dict()
    for subject, predictions_and_truths in subjects_predictions.items():
        # Verify that the ground truth is the same
        # todo: not necessarily true in self-supervised learning
        all_ground_truths = tuple(yyhat.y_true for yyhat in predictions_and_truths)
        if not all(torch.equal(all_ground_truths[0], ground_truth) for ground_truth in all_ground_truths):
            raise ValueError("Expected all ground truths to be the same per subject, but that was not the case")

        # Set prediction to the average of all predictions, and the ground truth to the only element in the set
        _pred = torch.mean(torch.cat([torch.unsqueeze(yyhat.y_pred, dim=0)
                                      for yyhat in predictions_and_truths], dim=0), dim=0, keepdim=True)
        _true = all_ground_truths[0]
        subjects_pred_and_true[subject] = YYhat(y_pred=_pred, y_true=_true)

    # Get as torch tensors
    all_y_pred = torch.cat([yyhat.y_pred for yyhat in subjects_pred_and_true.values()], dim=0)
    all_y_true = torch.cat([torch.unsqueeze(yyhat.y_true, dim=0) if yyhat.y_true.dim() == 0 else yyhat.y_true
                            for yyhat in subjects_pred_and_true.values()], dim=0)

    # Maybe remove redundant dimension
    all_y_pred = torch.squeeze(all_y_pred, dim=-1)

    # Return predictions, truths, and subjects (i-th prediction and truth comes from the i-th subject)
    return all_y_pred, all_y_true, tuple(subjects_pred_and_true.keys())


def save_discriminator_histories_plots(path, histories):
    """
    Function for saving domain discriminator histories plots

    Parameters
    ----------
    path : str
    histories : Histories | tuple[Histories, ...]

    Returns
    -------
    None
    """
    # Maybe just convert to tuple
    if not isinstance(histories, tuple):
        histories = (histories,)

    # Quick input check
    if not all(isinstance(history, Histories) for history in histories):
        raise TypeError(f"Expected all histories to be of type 'Histories', but found "
                        f"{set(history for history in histories)}")

    # ----------------
    # Loop through all metrics
    # ----------------
    # Get all available metrics
    _all_metrics = []
    for history in histories:
        _all_metrics.extend(list(history.history.keys()))
    all_metrics = set(_all_metrics)  # Keep unique ones only

    for metric in all_metrics:
        pyplot.figure(figsize=(12, 6))

        for history in histories:
            pyplot.plot(range(1, len(history.history[metric]) + 1), history.history[metric], label=history.name)

        # ------------
        # Plot cosmetics
        # ------------
        font_size = 15

        pyplot.title(f"Performance ({metric.capitalize()})", fontsize=font_size + 5)
        pyplot.xlabel("Epoch", fontsize=font_size)
        pyplot.ylabel(metric.capitalize(), fontsize=font_size)
        pyplot.tick_params(labelsize=font_size)
        pyplot.legend(fontsize=font_size)
        pyplot.grid()

        # Save figure and close it
        pyplot.savefig(os.path.join(path, f"discriminator_{metric}.png"))

        pyplot.close()


def save_test_histories_plots(path, histories):
    """
    Save histories

    todo: now we have two...
    Parameters
    ----------
    path : str | pathlib.Path
    histories : dict[str, Histories]

    Returns
    -------
    None
    """
    # ----------------
    # Loop through all metrics
    # ----------------
    # Get all available metrics
    _all_metrics: Tuple[str, ...] = ()
    for name, history in histories.items():
        if not _all_metrics:
            _all_metrics += tuple(history.history.keys())
        else:
            if set(_all_metrics) != set(history.history.keys()):
                raise RuntimeError("Expected all metrics to be the same, but that was not the case")
    all_metrics = tuple(_all_metrics)

    for metric in all_metrics:
        pyplot.figure(figsize=(12, 6))

        for name, history in histories.items():
            pyplot.plot(range(1, len(history.history[metric]) + 1), history.history[metric], label=name)

        # ------------
        # Plot cosmetics
        # ------------
        font_size = 15

        pyplot.title(f"Test performance ({metric.capitalize()})", fontsize=font_size + 5)
        pyplot.xlabel("Epoch", fontsize=font_size)
        pyplot.ylabel(metric.capitalize(), fontsize=font_size)
        pyplot.tick_params(labelsize=font_size)
        pyplot.legend(fontsize=font_size)
        pyplot.grid()

        # Save figure and close it
        pyplot.savefig(os.path.join(path, f"{metric}.png"))
        pyplot.close()


def save_histories_plots(path, *, train_history=None, val_history=None, test_history=None, test_estimate=None):
    """
    Function for saving histories plots

    Parameters
    ----------
    path : str
    train_history : Histories
    val_history : Histories
    test_history : Histories
    test_estimate : Histories

    Returns
    -------
    None
    """
    # If no history object is passed, a warning is raised and None is returned (better to do nothing than potentially
    # ruin an experiment with an unnecessary error)
    if all(history is None for history in (train_history, val_history, test_history, test_estimate)):
        warnings.warn("No history object was passed, skip saving histories plots...", PlotNotSavedWarning)
        return

    # ----------------
    # Loop through all metrics
    # ----------------
    # Get all available metrics
    _all_metrics: Tuple[str, ...] = ()
    if train_history is not None:
        _all_metrics += tuple(train_history.history.keys())
    if val_history is not None:
        _all_metrics += tuple(val_history.history.keys())
    if test_history is not None:
        _all_metrics += tuple(test_history.history.keys())
    if test_estimate is not None:
        _all_metrics += tuple(test_estimate.history.keys())
    all_metrics = set(_all_metrics)

    for metric in all_metrics:
        pyplot.figure(figsize=(12, 6))

        # Maybe plot training history
        if train_history is not None:
            pyplot.plot(range(1, len(train_history.history[metric]) + 1), train_history.history[metric],
                        label="Train", color="blue")

        # Maybe plot validation history
        if val_history is not None:
            pyplot.plot(range(1, len(val_history.history[metric]) + 1), val_history.history[metric], label="Validation",
                        color="orange")

        # Maybe plot validation history
        if test_history is not None:
            pyplot.plot(range(1, len(test_history.history[metric]) + 1), test_history.history[metric],
                        label="Test", color="green")

        # Maybe plot test history
        if test_estimate is not None:
            # The test estimate metric will just be a line across the figure. Need to get stop x value
            # Start value
            x_max = []
            if train_history is not None:
                x_max.append(len(train_history.history[metric]))
            if val_history is not None:
                x_max.append(len(val_history.history[metric]))
            if test_history is not None:
                x_max.append(len(test_history.history[metric]))
            x_stop = max(x_max) if x_max else 2

            # Plot
            pyplot.plot((1, x_stop), (test_estimate.history[metric], test_estimate.history[metric]),
                        label="Test estimate", color="red")

        # ------------
        # Plot cosmetics
        # ------------
        font_size = 15

        pyplot.title(f"Performance ({metric.capitalize()})", fontsize=font_size+5)
        pyplot.xlabel("Epoch", fontsize=font_size)
        pyplot.ylabel(metric.capitalize(), fontsize=font_size)
        pyplot.tick_params(labelsize=font_size)
        pyplot.legend(fontsize=font_size)
        pyplot.grid()

        # Save figure and close it
        pyplot.savefig(os.path.join(path, f"{metric}.png"))

        pyplot.close()


def is_improved_model(old_metrics, new_metrics, main_metric):
    """
    Function for checking if the new set of metrics is evaluated as better than the old metrics, defined by a main
    metric

    Parameters
    ----------
    old_metrics : dict[str, float] | None
    new_metrics : dict[str, float]
    main_metric : str

    Returns
    -------
    bool

    Examples
    --------
    >>> my_old_metrics = {"mae": 3, "mse": 7.7, "mape": 0.3, "pearson_r": 0.9, "spearman_rho": 0.8, "r2_score": -3.1}
    >>> my_new_metrics = {"mae": 3.2, "mse": 4.4, "mape": 0.2, "pearson_r": 0.7, "spearman_rho": 0.9, "r2_score": -3.05}
    >>> is_improved_model(None, my_new_metrics, main_metric="mae")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="mae")
    False
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="mse")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="mape")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="pearson_r")
    False
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="spearman_rho")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="r2_score")
    True
    >>> is_improved_model(my_old_metrics, my_new_metrics, main_metric="not_a_metric")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Expected the metric to be in (...), but found 'not_a_metric'
    """
    # If the old metrics is None, it means that this is the first epoch
    if old_metrics is None:
        return True

    # ----------------
    # Evaluate
    # ----------------
    if higher_is_better(main_metric):
        return old_metrics[main_metric] < new_metrics[main_metric]
    else:
        return old_metrics[main_metric] > new_metrics[main_metric]
