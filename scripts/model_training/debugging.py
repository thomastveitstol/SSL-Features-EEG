import random

from elecssl.data.paths import get_results_dir
from elecssl.models.experiments.hpo_experiment import SimpleElecsslHPO, MultivariableElecsslHPO, AllHPOExperiments, \
    PretrainHPO


def main():
    # ----------------
    # Continue experiment
    # ----------------
    path = get_results_dir() / "experiments_2025-03-15_145839"

    runs_for_integrity_check = (
        MultivariableElecsslHPO.load_previous(path=path / "multivariable_elecssl"),
        SimpleElecsslHPO.load_previous(path=path / "simple_elecssl"),
        PretrainHPO.load_previous(path=path / "pretraining"),
        )
    AllHPOExperiments.verify_test_set_integrity(runs_for_integrity_check)
    # print(experiment.get_trials_and_folders(completed_only=True))


if __name__ == "__main__":
    main()
