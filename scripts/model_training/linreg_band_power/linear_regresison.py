"""
A linear regression baseline
"""
import itertools
from typing import Literal, Tuple

import numpy
import pandas
from sklearn.linear_model import LinearRegression

from elecssl.data.datasets.getter import get_dataset
from elecssl.data.subject_split import get_data_split, Subject, subjects_tuple_to_dict
from elecssl.models.ml_models.ml_model_base import MLModel


def _build_features_matrix(sample_sizes, features, target_variable, return_subjects):
    """
    Function for building a data matrix

    Parameters
    ----------
    sample_sizes . dict[str, str | int]
    features : tuple[str, ...]
        All 'features' must be loadable from .load_targets()
    target_variable : str
        The target variable. Will be renamed to 'clinical_target' in the returned dataframe

    Returns
    -------
    pandas.DataFrame
    """
    # Initialise data matrix
    data_matrix = {"subject": [], "clinical_target": [], **{feature: [] for feature in features}}
    all_subjects = []

    # Loop though all datasets to use
    for dataset_name, sample_size in sample_sizes.items():
        # Get the dataset
        dataset = get_dataset(dataset_name=dataset_name)

        # Get the subject IDs
        if sample_size == "all":
            subjects = dataset.get_subject_ids()
        else:
            subjects = dataset.get_subject_ids()[:sample_size]

        # Update data matrix
        dataset_subjects =[Subject(subject_id=subject, dataset_name=dataset_name) for subject in subjects]
        all_subjects.extend(dataset_subjects)
        for feature in features:
            feature_array = dataset.load_targets(target=feature)
            data_matrix[feature].extend(feature_array)
        target_array = dataset.load_targets(target=target_variable)
        data_matrix["clinical_target"].extend(target_array)

    # Make dataframe, fix index, and return
    data_matrix["subject"] = all_subjects
    df = pandas.DataFrame(data_matrix)
    df.set_index("subject", inplace=True)

    if return_subjects:
        return df, tuple(all_subjects)
    else:
        return df


def _fit_and_compute_residuals(x, y):
    model: LinearRegression = LinearRegression().fit(X=numpy.expand_dims(x.to_numpy(), axis=-1), y=y)  # type: ignore
    return y - model.predict(numpy.expand_dims(x.to_numpy(), axis=-1))


def _get_residuals(df, in_out_features: Tuple[Tuple[str, str], ...]):
    data_matrix = dict()
    for x_name, y_name in in_out_features:
        data_matrix[f"{x_name} -> {y_name}"] = _fit_and_compute_residuals(x=df[x_name], y=df[y_name])
    return pandas.DataFrame(data_matrix)


def _compute_deviation_from_expectation(df, difference):
    feature_names = df.drop(labels='clinical_target', inplace=False, axis='columns').columns
    if difference == "eoec":
        # Get feature names
        eo_features = tuple(feature for feature in feature_names if feature.endswith("_eo"))
        ec_features = tuple(f"{eo_feature[:-2]}ec" for eo_feature in eo_features)

        # Get data matrix of residuals
        data_matrix = _get_residuals(df, in_out_features=tuple(zip(eo_features, ec_features)))

    elif difference == "eceo":
        # Get feature names
        ec_features = tuple(feature for feature in feature_names if feature.endswith("_ec"))
        eo_features = tuple(f"{ec_feature[:-2]}eo" for ec_feature in ec_features)

        # Get data matrix of residuals
        data_matrix = _get_residuals(df, in_out_features=tuple(zip(eo_features, ec_features)))

    elif difference == "both":
        raise NotImplementedError  # todo: continue implementing...
    else:
        raise ValueError(f"Unexpected method for computing feature differences: {difference}")

    data_matrix["clinical_target"] = df["clinical_target"]
    return data_matrix


def main():
    # --------------
    # Choices
    # --------------
    sample_sizes = {"LEMON": "all", "DortmundVital": "all"}
    ocular_states = ("eo", "ec")
    freq_bands = ("delta", "theta", "alpha", "beta", "gamma")
    target = "age"
    evaluation_metric = "pearson_r"
    aggregation_method = "median"
    val_split = 0.2
    num_random_splits = 50
    seed = 42
    in_out: Literal[None, "eoec", "eceo", "both"] = "eceo"

    # Prepare the feature names
    features = tuple(f"log10_band_power_{freq_band}_{ocular_state}" for freq_band, ocular_state
                     in itertools.product(freq_bands, ocular_states))

    # --------------
    # Dataset preparation
    # --------------
    # Build feature matrix
    df, subjects = _build_features_matrix(sample_sizes=sample_sizes, features=features, target_variable=target,
                                          return_subjects=True)
    tot_num_subject = df.shape[0]
    df.dropna(inplace=True)
    subjects = [subject for subject in subjects if subject in df.index]

    # Maybe use the deviation from expectation
    if in_out is not None:
        df = _compute_deviation_from_expectation(df, difference=in_out)

    # Set the data split
    splits = get_data_split(split="RandomSplitsTV", val_split=val_split, num_random_splits=num_random_splits, seed=seed,
                            dataset_subjects=subjects_tuple_to_dict(subjects)).splits

    # Do some printing
    print(df)
    print(f"\nUsing features {tuple(df.drop(labels='clinical_target', inplace=False, axis='columns').columns)}")
    print(f"\nDropped {tot_num_subject - df.shape[0]} subjects due to NaN values")

    # --------------
    # Linear regression
    # --------------
    # Create model
    ml_model = MLModel(model="LinearRegression", model_kwargs=dict(), splits=splits, evaluation_metric=evaluation_metric,
                       aggregation_method=aggregation_method)

    # Evaluate performance
    score = ml_model.evaluate_features(non_test_df=df)

    print(f"\nObtained {aggregation_method} {evaluation_metric}: {score:.4f}")


if __name__ == "__main__":
    main()
