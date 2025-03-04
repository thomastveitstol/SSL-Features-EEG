from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import ElecsslHPO


def main():
    # ----------------
    # Continue experiment
    # ----------------
    path = get_results_dir() / "elecssl" / "elecssl_hpo_experiment_2025-03-03_183236"
    with ElecsslHPO.load_previous(path=path) as experiment:
        experiment.continue_hyperparameter_optimisation(path=path, num_trials=2)

if __name__ == "__main__":
    main()
