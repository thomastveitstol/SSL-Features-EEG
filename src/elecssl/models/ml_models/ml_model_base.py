from copy import deepcopy
from typing import List

import numpy
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from elecssl.models.metrics import Histories


class MLModel:
    """
    Main ML model. ML models are in this project intended to be used on the 'learned' biomarkers. Must be able to
    provide a feedback to the HPO algorithm and perform test evaluation
    """

    __slots__ = ("_ml_model", "_splits", "_evaluation_metric", "_aggregation_method")

    def __init__(self, *, model, model_kwargs, splits, evaluation_metric, aggregation_method):
        self._ml_model = _get_ml_model(model, **model_kwargs)
        self._splits = splits
        self._evaluation_metric = evaluation_metric
        self._aggregation_method = aggregation_method

    def evaluate_features(self, non_test_df: pandas.DataFrame):
        scores: List[float] = []
        for train, val, test in self._splits:
            # I'll reuse the splits as in the DL models, but I shouldn't actually have test set here
            assert not test, f"Expected test set to be empty, but found (N={len(test)}): {test}"

            # --------------
            # Model training
            # --------------
            # Make a copy of the model
            model = deepcopy(self._ml_model)

            # Get the features
            train_df = non_test_df.loc[train]

            # Fit it
            y_true = model.fit(
                X=train_df.drop(labels="clinical_target", inplace=False, axis="columns"), y=train_df["clinical_target"]
            )

            # Compute evaluation score
            y_pred = val_df = non_test_df.loc[val]
            model.predict(X=val_df.drop(labels="clinical_target", inplace=False, axis="columns"))
            score = Histories.compute_metric(metric=self._evaluation_metric, y_pred=y_pred, y_true=y_true)

            # Add it to the results
            scores.append(score)

        # Aggregate the scores to a single score
        return _aggregate_scores(method=self._aggregation_method, scores=scores)


# ------------
# Functions
# ------------
def _aggregate_scores(method: str, scores: List[float]):
    if method == "mean":
        return numpy.mean(scores)
    elif method == "median":
        return numpy.median(scores)
    else:
        raise ValueError(f"Method for aggregating the scores not recognised: {method}")


def _get_ml_model(model, **kwargs):
    # All available ML models must be included here
    available_models = (DecisionTreeRegressor, DecisionTreeClassifier, LinearRegression)

    # Loop through and select the correct one
    for available_model in available_models:
        if model == available_model.__name__:
            return available_model(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The ML model '{model}' was not recognised. Please select among the following: "
                     f"{tuple(m.__name__ for m in available_models)}")
