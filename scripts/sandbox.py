import os.path
from pathlib import Path

import ConfigSpace
import pandas
import seaborn
import torch
from matplotlib import pyplot
from optuna.samplers import TPESampler

from elecssl.data.paths import get_results_dir
from elecssl.data.results_analysis.utils import load_hpo_study
from elecssl.models.experiments.hpo_experiment import PredictionModelsHPO, PretrainHPO, ElecsslHPO
from elecssl.models.mts_modules.green_model import GreenModel


def get_config_space():
    results_dir = get_results_dir()
    study_paths = (results_dir / "elecssl" / "elecssl_hpo_experiment_2025-02-06_154222",
                   results_dir / "pretraining" / "pretraining_hpo_experiment_2025-02-07_030752",
                   results_dir / "prediction_models" / "prediction_models_hpo_experiment_2025-02-07_064413")
    study_paths = (results_dir / "prediction_models" / "prediction_models_hpo_experiment_2025-02-07_064413",)

    for study_path in study_paths:
        study = load_hpo_study(name=f"{study_path.parts[-2]}-study", path=study_path)
        print(study.get_trials()[7])


def plot_studies():
    results_dir = get_results_dir()
    study_paths = (results_dir /  "elecssl" / "elecssl_hpo_experiment_2025-02-06_154222",
                   results_dir /  "pretraining" / "pretraining_hpo_experiment_2025-02-07_030752",
                   results_dir /  "prediction_models" / "prediction_models_hpo_experiment_2025-02-07_064413")

    for study_path in study_paths:
        study = load_hpo_study(name=f"{study_path.parts[-2]}-study", path=study_path)

        scores = tuple(trial.value for trial in study.trials)
        trial_number = tuple(trial.number for trial in study.trials)

        seaborn.scatterplot(x=trial_number, y=scores)
    pyplot.show()


def make_boxplots():
    # ---------------
    # Some design choices
    # ---------------
    # How to compute and aggregate
    selection_metric = "r2_score"
    target_metrics = ("pearson_r", "spearman_rho", "r2_score", "mae", "mse", "mape")
    method = "mean"

    # Plot details
    x = "test_r2_score"
    y = "experiment_name"

    # Other stuff
    results_dir = get_results_dir()

    # ---------------
    # Analysis
    # ---------------
    # Generate dataframe
    prediction_models_df = PredictionModelsHPO.generate_test_scores_df(
        path=results_dir / "prediction_models" / "prediction_models_hpo_experiment_2025-02-07_064413",
        selection_metric=selection_metric, target_metrics=target_metrics, method=method, include_experiment_name=True
    )
    pretraining_df = PretrainHPO.generate_test_scores_df(
        path=results_dir / "pretraining" / "pretraining_hpo_experiment_2025-02-07_030752",
        selection_metric=selection_metric, target_metrics=target_metrics, method=method, include_experiment_name=True
    )
    df = pandas.concat((prediction_models_df, pretraining_df), axis="rows")
    df.reset_index(inplace=True)

    # Plotting
    seaborn.boxplot(data=df, x=x, y=y, linewidth=1.2, dodge=True, showfliers=False, fill=False)
    seaborn.stripplot(data=df, x=x, y=y, jitter=True, dodge=True, size=3, alpha=0.5, marker='o')

    pyplot.show()


def check_test_set_integrity():
    path = get_results_dir() / "prediction_models" / "debug_prediction_models_hpo_experiment_2025-02-27_154700"
    PredictionModelsHPO.verify_test_set_integrity(path=path)

    path = get_results_dir() / "elecssl" / "debug_elecssl_hpo_experiment_2025-02-28_151005"
    ElecsslHPO.verify_test_set_integrity(path=path)


def make_hue_boxplots_single_study():
    # ---------------
    # Some design choices
    # ---------------
    # How to compute and aggregate
    selection_metric = "r2_score"
    target_metrics = ("pearson_r", "spearman_rho", "r2_score", "mae", "mse", "mape")
    method = "mean"
    y = "architecture"
    hue = "spatial_dimension_handling"

    # Plot details
    x = "test_r2_score"

    # x, y = y, x

    # Other stuff
    results_dir = get_results_dir() / "prediction_models" / "debug_prediction_models_hpo_experiment_2025-02-27_154700"

    # ---------------
    # Analysis
    # ---------------
    # Generate dataframe
    df = PredictionModelsHPO.generate_test_scores_df(
        path=results_dir, selection_metric=selection_metric, target_metrics=target_metrics, method=method,
        include_experiment_name=False
    )

    # Add HPCs
    df = PredictionModelsHPO.add_hp_configurations_to_dataframe(df, hps=(y, hue), results_dir=results_dir)

    # Plotting
    # seaborn.scatterplot(data=df, x=x, y=y)
    seaborn.boxplot(data=df, x=x, y=y, hue=hue, linewidth=1.2, dodge=True, showfliers=False, fill=False)
    seaborn.stripplot(data=df, x=x, y=y, hue=hue, jitter=True, dodge=True, size=3, alpha=0.5, marker='o')
    # pyplot.xscale("log")

    pyplot.show()


def try_making_config_space_json():
    # Create a configuration space
    cs = ConfigSpace.ConfigurationSpace({
        "lr": ConfigSpace.UniformFloatHyperparameter("lr", lower=0.001, upper=0.1, log=True),
        "bs": ConfigSpace.UniformIntegerHyperparameter("batch_size", lower=16, upper=128),
    })

    # Save to a JSON file
    path = Path(os.path.dirname(__file__), "config_space.json")
    cs.to_json(path=path)

    print(ConfigSpace.ConfigurationSpace.from_json(path))


def save_green():
    kwargs = {"bi_out": 28,
              "orth_weights": True,

              "conv_stride": 3,
              "dropout": 0.4166762566834736,
              "hidden_dim": [11],
              "kernel_width_s": 3.5051339860376958,
              "logref": "logeuclid",
              "n_freqs": 48,
              "oct_max": 6.228701883563893,
              "oct_min": 0.6851740516683408,
              "pool_layer": "RealCovariance",
              "pool_layer_kwargs": {},
              "random_f_init": False,
              "reg": 6.0727362401454394e-06,
              "shrinkage_init": -2.67768716062845
              }
    torch.save(GreenModel(in_channels=5, num_classes=1, sampling_freq=90, **kwargs),
               Path(os.path.dirname(__file__)) / "green_model.pt")


def main():
    check_test_set_integrity()
    # make_hue_boxplots_single_study()


if __name__ == "__main__":
    main()