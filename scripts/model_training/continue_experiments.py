"""
Script which can be used to resume experiments. Hopefully it won't be used though...
"""
from pathlib import Path

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import AllHPOExperiments


def main():
    loading_experiment_path = get_results_dir() / "experiments_2025-03-25_183139"
    with AllHPOExperiments.load_previous(loading_experiment_path) as experiments:
        experiments.continue_simple_elecssl_hpo(num_trials=None)
        experiments.continue_multivariable_elecssl_hpo(num_trials=None)


if __name__ == "__main__":
    main()
