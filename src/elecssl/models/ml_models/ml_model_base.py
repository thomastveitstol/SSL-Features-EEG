from copy import deepcopy
from typing import List, Dict

import numpy
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LassoLars, Lars, BayesianRidge, \
    ARDRegression, OrthogonalMatchingPursuit
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
    >>> cols_ = ['clinical_target'] + [f'var{i}' for i in range(1, 4)]  # type: ignore
    >>> numpy.random.seed(42)
    >>> my_df = pandas.DataFrame(numpy.random.rand(len(idxs_), len(cols_)), index=idxs_, columns=cols_).round(2)
    >>> my_df.head()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
                                                 clinical_target  var1  var2  var3
    Subject(subject_id='S1', dataset_name='D1')             0.37  0.95  0.73  0.60
    Subject(subject_id='S2', dataset_name='D1')             0.16  0.16  0.06  0.87
    Subject(subject_id='S1', dataset_name='D2')             0.60  0.71  0.02  0.97
    Subject(subject_id='S2', dataset_name='D2')             0.83  0.21  0.18  0.18
    Subject(subject_id='S3', dataset_name='D2')             0.30  0.52  0.43  0.29
    >>> my_model = MLModel(model="LinearRegression", model_kwargs={}, splits=my_splits, evaluation_metric="mae",
    ...                    aggregation_method="mean")
    >>> float(round(my_model.evaluate_features(non_test_df=my_df), 3))
    0.191
    >>> my_model = MLModel(model="Lasso", model_kwargs={"alpha": 0.5}, splits=my_splits, evaluation_metric="mae",
    ...                    aggregation_method="mean")
    >>> float(round(my_model.evaluate_features(non_test_df=my_df), 3))
    0.212
    >>> my_model = MLModel(model="ElasticNet", model_kwargs={"alpha": 0.5, "l1_ratio": 0.3}, splits=my_splits,
    ... evaluation_metric="mae", aggregation_method="mean")
    >>> float(round(my_model.evaluate_features(non_test_df=my_df), 3))
    0.212
    >>> my_model = MLModel(model="Ridge", model_kwargs={"alpha": 0.5}, splits=my_splits, evaluation_metric="mae",
    ...                    aggregation_method="mean")
    >>> float(round(my_model.evaluate_features(non_test_df=my_df), 3))
    0.201
    >>> my_model = MLModel(model="Lars", model_kwargs={"n_nonzero_coefs": 4}, splits=my_splits, evaluation_metric="mae",
    ...                    aggregation_method="mean")
    >>> float(round(my_model.evaluate_features(non_test_df=my_df), 3))
    0.191
    >>> my_model = MLModel(model="LassoLars", model_kwargs={"alpha": 0.5}, splits=my_splits, evaluation_metric="mae",
    ...                    aggregation_method="mean")
    >>> float(round(my_model.evaluate_features(non_test_df=my_df), 3))
    0.212
    >>> my_model = MLModel(model="OrthogonalMatchingPursuit", model_kwargs={}, splits=my_splits,
    ...                    evaluation_metric="mae", aggregation_method="mean")
    >>> float(round(my_model.evaluate_features(non_test_df=my_df), 3))
    0.273
    >>> my_model = MLModel(model="BayesianRidge", model_kwargs={}, splits=my_splits, evaluation_metric="mae",
    ...                    aggregation_method="mean")
    >>> float(round(my_model.evaluate_features(non_test_df=my_df), 3))
    0.22
    >>> my_model = MLModel(model="ARDRegression", model_kwargs={}, splits=my_splits, evaluation_metric="mae",
    ...                    aggregation_method="mean")
    >>> float(round(my_model.evaluate_features(non_test_df=my_df), 3))
    0.209
    """

    __slots__ = ("_ml_model", "_splits", "_evaluation_metric", "_aggregation_method", "_fitted_ml_models")

    def __init__(self, *, model, model_kwargs, splits, evaluation_metric, aggregation_method):
        self._ml_model = _get_ml_model(model, **model_kwargs)
        self._fitted_ml_models = []
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
        # If models have already been fit, you should create a new object instead
        if self._fitted_ml_models:
            raise RuntimeError("Tried to evaluate features (which includes training ML models), but the ML models are "
                               "already fitted")

        scores: List[float] = []
        for train, val, test in self._splits:
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

            # Store the model
            self._fitted_ml_models.append(model)

        # Aggregate the scores to a single score
        return _aggregate_scores(method=self._aggregation_method, scores=scores)

    def predict_and_score(self, df, metrics, aggregation_method):
        # Maybe set the metrics to a pre-defined set
        if isinstance(metrics, str):
            metrics = Histories.get_default_metrics(metrics=metrics)

        # Make predictions for all the available models. The i-th prediction (at axis 0) is from the i-th model
        input_data = df.drop(labels="clinical_target", inplace=False, axis="columns")
        predictions = numpy.concatenate([numpy.expand_dims(model.predict(X=input_data), axis=0)
                                         for model in self._fitted_ml_models], axis=0)

        # Aggregate the predictions
        aggregated_predictions = _aggregate_predictions(method=aggregation_method, predictions=predictions)

        # Compute the scores
        target_data = df["clinical_target"].to_numpy()
        performance: Dict[str, float] = {}
        for metric in metrics:
            performance[metric] = Histories.compute_metric(metric, y_pred=aggregated_predictions, y_true=target_data)

        return aggregated_predictions, performance


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


def _aggregate_predictions(method, predictions):
    """
    Function for aggregating multiple predictions by, e.g., computing the mean. The i-th prediction (at axis=0) should
    be from the i-th ML model

    Parameters
    ----------
    method : str
    predictions : numpy.ndarray

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> my_agg = _aggregate_predictions("mean", numpy.random.rand(4, 5, 6))
    >>> my_agg.shape, type(my_agg)  # type: ignore
    ((5, 6), <class 'numpy.ndarray'>)
    """
    if method == "mean":
        return numpy.mean(predictions, axis=0)  # type: ignore
    elif method == "median":
        return numpy.median(predictions, axis=0)  # type: ignore
    else:
        raise ValueError(f"Method for aggregating the predictions not recognised: {method}")


def _get_ml_model(model, **kwargs):
    # All available ML models must be included here
    available_models = (DecisionTreeRegressor, DecisionTreeClassifier, LinearRegression, Lasso, Ridge, ElasticNet, Lars,
                        LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression)

    # Loop through and select the correct one
    for available_model in available_models:
        if model == available_model.__name__:
            return available_model(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The ML model '{model}' was not recognised. Please select among the following: "
                     f"{tuple(m.__name__ for m in available_models)}")
