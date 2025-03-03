from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import PredictionModelsHPO


def main():
    # ----------------
    # Continue experiment
    # ----------------
    path = get_results_dir() / "prediction_models" / "debug_prediction_models_hpo_experiment_2025-02-28_155939"
    prediction_models_experiment = PredictionModelsHPO.load_previous(path=path)
    with prediction_models_experiment as experiment:
        experiment.continue_hyperparameter_optimisation(path=path, num_trials=4)

if __name__ == "__main__":
    main()
