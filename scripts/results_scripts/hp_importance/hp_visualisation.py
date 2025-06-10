from pathlib import Path

import optuna
import pandas
import seaborn
from matplotlib import pyplot
from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution

from elecssl.data.paths import get_results_dir


CHP = CategoricalDistribution
FHP = FloatDistribution
IHP = IntDistribution


_PRETTY_NAME = {
    "GreenModel": "Green",
    "InceptionNetwork": "Inception N.",
    "ShallowFBCSPNetMTS": "Shallow N.",
    "Deep4NetMTS": "Deep N.",
    "TCNMTS": "TCN",
    "DortmundVital": "DV",
    "Wang+DortmundVital": "Wang+DV"
}
_NUM_TO_STR = ("input_length",)


def main():
    hp_1 = "out_freq_band"
    hp_2 = "normalisation"

    study_name = "simple_elecssl"
    random_only = False

    experiment_time = "2025-05-20_141517"

    experiment_name = f"experiments_{experiment_time}"
    experiments_path = Path(get_results_dir() / experiment_name)

    value_name = "Performance"
    val_lim = (-0.3, 1)

    # -------------
    # Make plots
    # -------------
    study_path = (experiments_path / study_name / f"{study_name}-study.db")
    study_storage = f"sqlite:///{study_path}"

    # Load HPs from optuna study object
    study = optuna.load_study(study_name=f"{study_name}-study", storage=study_storage)
    trials_df: pandas.DataFrame = study.trials_dataframe()

    if random_only:
        trials_df = trials_df[trials_df["user_attrs_trial_sampler"] != "TPESampler"]

    # Remove annoying prefix
    _prefix = "params_"
    col_mapping = {col: col[len(_prefix):] for col in trials_df.columns if col.startswith(_prefix)}
    trials_df.rename(columns=col_mapping, inplace=True)

    # Cast some columns to string
    for col in _NUM_TO_STR:
        trials_df[col] = trials_df[col].astype(str)

    # Replace ugly names with pretty ones
    old_hp_1 = hp_1
    old_hp_2 = hp_2
    hp_1 = hp_1.replace("_", " ").capitalize()
    hp_2 = hp_2.replace("_", " ").capitalize()

    trials_df.replace(_PRETTY_NAME, inplace=True)
    trials_df.rename(columns={old_hp_1: hp_1, old_hp_2: hp_2, "value": value_name}, inplace=True)

    # Filter on non-NaN
    mask = ~(trials_df[value_name].isnull() | trials_df[hp_1].isnull() | trials_df[hp_2].isnull())
    trials_df = trials_df[mask]

    # Make plot based on the type
    distributions = study.trials[-1].distributions

    hp_type_1 = distributions[old_hp_1]
    hp_type_2 = distributions[old_hp_2]
    if isinstance(hp_type_1, CHP) and isinstance(hp_type_2, CHP):
        seaborn.boxplot(trials_df, x=value_name, y=hp_1, hue=hp_2, linewidth=1.2, showfliers=False, fill=False,
                        dodge=True)
        seaborn.stripplot(trials_df, x=value_name, y=hp_1, hue=hp_2, dodge=True, jitter=True, size=3, marker='o',
                          alpha=0.5, legend=False)
        pyplot.xlim(*val_lim)

        # Theme (shading with grey)
        for i, _ in enumerate(set(trials_df[hp_1])):
            if i % 2 == 0:  # Shade alternate categories
                pyplot.axhspan(i - 0.5, i + 0.5, color="lightgrey", alpha=0.5)

    elif isinstance(hp_type_1, CHP) and isinstance(hp_type_2, (FHP, IHP)):
        log_x = hp_type_2.log
        for category in set(trials_df[hp_1]):
            seaborn.regplot(trials_df[trials_df[hp_1] == category], x=hp_2, y=value_name, label=category, logx=log_x)
        if log_x:
            pyplot.xscale("log")
        pyplot.ylim(*val_lim)
        pyplot.legend()
    elif isinstance(hp_type_2, CHP) and isinstance(hp_type_1, (FHP, IHP)):
        log_x = hp_type_1.log
        for category in set(trials_df[hp_2]):
            seaborn.regplot(trials_df[trials_df[hp_2] == category], x=hp_1, y=value_name, label=category, logx=log_x)
        if log_x:
            pyplot.xscale("log")
        pyplot.ylim(*val_lim)
        pyplot.legend()
    elif isinstance(hp_type_1, (FHP, IHP)) and isinstance(hp_type_2, (FHP, IHP)):
        log_x = hp_type_1.log
        log_y = hp_type_2.log
        seaborn.scatterplot(trials_df[trials_df[value_name] > 0], x=hp_1, y=hp_2, hue=value_name, palette='viridis')
        if log_x:
            pyplot.xscale("log")
        if log_y:
            pyplot.yscale("log")
        # pyplot.ylim(*val_lim)
        pyplot.legend()
    else:
        raise NotImplementedError
    pyplot.grid()
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":
    main()
