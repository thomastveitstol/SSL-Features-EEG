from copy import deepcopy
from typing import List

import numpy
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from elecssl.models.metrics import Histories


class MLModel:
    """
    Main ML model. ML models are in this project intended to be used on the 'learned' biomarkers. Must be able to
    provide a feedback to the HPO algorithm and perform test evaluation.

    Examples
    --------
    >>> import pandas
    >>> from elecssl.data.subject_split import RandomSplitsTV, Subject
    >>> my_subjects = {"D1": ("S1", "S2"), "D2": ("S1", "S2", "S3"), "D3": ("P1", "P2", "P3", "P4"), "D4": ("P1", "P2")}
    >>> my_num_splits = 4
    >>> my_splits = RandomSplitsTV(my_subjects, val_split=0.2, num_random_splits=my_num_splits, seed=42).splits
    >>> idxs_ = [Subject(subject_id=s_id, dataset_name=d) for d, s in my_subjects.items() for s_id in s]  # type: ignore
    >>> cols_ = ['clinical_target'] + [f'var{i}' for i in range(1, 11)]  # type: ignore
    >>> numpy.random.seed(42)
    >>> my_df = pandas.DataFrame(numpy.random.rand(len(idxs_), len(cols_)), index=idxs_, columns=cols_).round(2)
    >>> my_df.head()
                                                 clinical_target  var1  ...  var9  var10
    Subject(subject_id='S1', dataset_name='D1')             0.37  0.95  ...  0.71   0.02
    Subject(subject_id='S2', dataset_name='D1')             0.97  0.83  ...  0.61   0.14
    Subject(subject_id='S1', dataset_name='D2')             0.29  0.37  ...  0.17   0.07
    Subject(subject_id='S2', dataset_name='D2')             0.95  0.97  ...  0.03   0.91
    Subject(subject_id='S3', dataset_name='D2')             0.26  0.66  ...  0.89   0.60
    <BLANKLINE>
    [5 rows x 11 columns]
    >>> my_model = MLModel(model="LinearRegression", model_kwargs={}, splits=my_splits, evaluation_metric="mae",
    ...                    aggregation_method="mean")
    >>> my_model.evaluate_features(non_test_df=my_df)  # doctest: +ELLIPSIS
    np.float64(0.578...)
    """

    __slots__ = ("_ml_model", "_splits", "_evaluation_metric", "_aggregation_method")

    def __init__(self, *, model, model_kwargs, splits, evaluation_metric, aggregation_method):
        self._ml_model = _get_ml_model(model, **model_kwargs)
        self._splits = splits
        self._evaluation_metric = evaluation_metric
        self._aggregation_method = aggregation_method

    def evaluate_features(self, non_test_df):
        """
        Method for evaluating.

        Note that 'clinical_target' must be the name of the target variable. All other columns will be used as features

        Parameters
        ----------
        non_test_df : pandas.DataFrame

        Returns
        -------
        float
        """
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
            train_df = non_test_df.loc[list(train)]

            # Fit it
            model = model.fit(
                X=train_df.drop(labels="clinical_target", inplace=False, axis="columns"), y=train_df["clinical_target"]
            )

            # Compute evaluation score
            val_df = non_test_df.loc[list(val)]
            y_pred = model.predict(X=val_df.drop(labels="clinical_target", inplace=False, axis="columns"))
            score = Histories.compute_metric(
                metric=self._evaluation_metric, y_pred=y_pred, y_true=val_df["clinical_target"].to_numpy()
            )

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
