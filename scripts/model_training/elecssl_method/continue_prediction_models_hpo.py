from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import PredictionModelsHPO


def main():
    # ----------------
    # Continue experiment
    # ----------------
    path = get_results_dir() / "prediction_models" / "prediction_models_hpo_experiment_2025-03-04_012624"
    with PredictionModelsHPO.load_previous(path=path) as experiment:
        experiment.continue_hyperparameter_optimisation(path=path, num_trials=2)

if __name__ == "__main__":
    main()
