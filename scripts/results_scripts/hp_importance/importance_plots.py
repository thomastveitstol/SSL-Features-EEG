"""
Script for creating plots of hyperparameter importance.

I've only managed to compute interaction effects with an old environment. So, to use this, you need to use a venv with
fanova. This also requires pyrfr, which I was not able to download with Python 3.12
"""
import itertools
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import Any, List, Dict, Optional, Tuple

import ConfigSpace
import fanova  # type: ignore
import numpy
import optuna
import pandas
import seaborn
from matplotlib import pyplot

try:
    from elecssl.data.paths import get_results_dir
except ModuleNotFoundError:
    def get_results_dir():
        return Path("/home/thomas/PycharmProjects/SSL-Features-EEG/src/elecssl/data/results")


# ------------
# Updated classes
# ------------
class UpdatedFANOVA(fanova.fANOVA):
    """
    Just need make some changes to a method, original fANOVA (parent class) is found at
    https://github.com/automl/fanova/blob/master/fanova/fanova.py
    """

    def quantify_importance(self, dims):
        if type(dims[0]) == str:
            idx = []
            for i, param in enumerate(dims):
                idx.append(self.cs.get_idx_by_hyperparameter_name(param))
            dimensions = idx
        # make sure that all the V_U values are computed for each tree
        else:
            dimensions = dims

        self._fANOVA__compute_marginals(dimensions)

        importance_dict = {}

        for k in range(1, len(dimensions) + 1):
            for sub_dims in itertools.combinations(dimensions, k):
                if type(dims[0]) == str:
                    dim_names = []
                    for j, dim in enumerate(sub_dims):
                        dim_names.append(self.cs.get_hyperparameter_by_idx(dim))
                    dim_names = tuple(dim_names)
                    importance_dict[dim_names] = {}
                else:
                    importance_dict[sub_dims] = {}
                # clean here to catch zero variance in a trees
                non_zero_idx = numpy.nonzero([self.trees_total_variance[t] for t in range(self.n_trees)])
                if len(non_zero_idx[0]) == 0:
                    raise RuntimeError('Encountered zero total variance in all trees.')

                fractions_total = numpy.array([self.V_U_total[sub_dims][t] / self.trees_total_variance[t]
                                               for t in non_zero_idx[0]])
                fractions_individual = numpy.array([self.V_U_individual[sub_dims][t] / self.trees_total_variance[t]
                                                    for t in non_zero_idx[0]])

                if type(dims[0]) == str:
                    importance_dict[dim_names]['individual importance'] = numpy.mean(fractions_individual)
                    importance_dict[dim_names]['total importance'] = numpy.mean(fractions_total)
                    importance_dict[dim_names]['individual std'] = numpy.std(fractions_individual)
                    importance_dict[dim_names]['total std'] = numpy.std(fractions_total)
                    for tree, fraction in enumerate(fractions_individual):
                        importance_dict[dim_names][f"Tree {tree}"] = fraction

                else:
                    importance_dict[sub_dims]['individual importance'] = numpy.mean(fractions_individual)
                    importance_dict[sub_dims]['total importance'] = numpy.mean(fractions_total)
                    importance_dict[sub_dims]['individual std'] = numpy.std(fractions_individual)
                    importance_dict[sub_dims]['total std'] = numpy.std(fractions_total)
                    for tree, fraction in enumerate(fractions_individual):
                        importance_dict[sub_dims][f"Tree {tree}"] = fraction

        return importance_dict

    def get_most_important_pairwise_marginals(self, params=None, n=None):
        """
        Similar to base class, but returns all pairwise marginals and returns the standard
        deviations too.

        Most is taken from the original function, see:
        https://github.com/automl/fanova/blob/master/fanova/fanova.py
        """
        tot_imp_dict = OrderedDict()
        pairwise_marginals = []
        if params is None:
            dimensions = range(self.n_dims)
        else:
            if type(params[0]) == str:
                idx = []
                for i, param in enumerate(params):
                    idx.append(self.cs.get_idx_by_hyperparameter_name(param))
                dimensions = idx

            else:
                dimensions = params
        # pairs = it.combinations(dimensions,2)
        pairs = [x for x in itertools.combinations(dimensions, 2)]
        if params:
            n = len(list(pairs))

        try:
            from progressbar import progressbar  # type: ignore
            hp_pair_loop = progressbar(pairs, redirect_stdout=True, prefix="HP pairs ")
        except ImportError:
            hp_pair_loop = pairs
        for combi in hp_pair_loop:
            pairwise_marginal_performance = self.quantify_importance(combi)
            tot_imp = pairwise_marginal_performance[combi]['individual importance']  # Importance
            std = pairwise_marginal_performance[combi]['individual std']  # std (added)
            trees = {}
            for tree in range(self.n_trees):
                trees[f"Tree {tree}"] = pairwise_marginal_performance[combi][f"Tree {tree}"]
            combi_names = [self.cs_params[combi[0]].name, self.cs_params[combi[1]].name]  # HP names
            pairwise_marginals.append((tot_imp, std, trees, combi_names[0], combi_names[1]))

        pairwise_marginal_performance = sorted(pairwise_marginals, reverse=True)

        if n is None:
            for marginal, std, all_trees, p1, p2 in pairwise_marginal_performance:
                tot_imp_dict[(p1, p2)] = marginal, std, all_trees
        else:
            for marginal, std, all_trees, p1, p2 in pairwise_marginal_performance[:n]:
                tot_imp_dict[(p1, p2)] = marginal, std, all_trees
        self._dict = True

        return tot_imp_dict


def _optuna_to_configspace(distributions):
    """Convert Optuna distributions to ConfigSpace."""
    cs = ConfigSpace.ConfigurationSpace()

    for param, dist in distributions.items():
        if isinstance(dist, optuna.distributions.FloatDistribution):
            cs.add(ConfigSpace.UniformFloatHyperparameter(param, dist.low, dist.high, log=dist.log))
        elif isinstance(dist, optuna.distributions.IntDistribution):
            cs.add(ConfigSpace.UniformIntegerHyperparameter(param, dist.low, dist.high, log=dist.log))
        elif isinstance(dist, optuna.distributions.CategoricalDistribution):
            cs.add(ConfigSpace.CategoricalHyperparameter(param, dist.choices))
        else:
            raise ValueError(f"Unsupported distribution: {dist}")

    return cs


_NUMERICAL_ENCODING = MappingProxyType({
    "RegionBasedPooling": 0, "Interpolation": 1,
    "delta": 0, "theta": 1, "alpha": 2, "beta": 3, "gamma": 4, "all": 5,
    "k_1": 0, "k_2": 1, "k_3": 2, "k_4": 3,
    "InceptionNetwork": 0, "ShallowFBCSPNetMTS": 1, "Deep4NetMTS": 2, "TCNMTS": 3, "GreenModel": 4,
    "True": 0, "False": 1, True: 0, False: 1,
    "MultiMSMean": 0, "MultiMSSharedRocket": 1, "MultiMSSharedRocketHeadRegion": 2,
    "CentroidPolygons": 0,
    "MSELoss": 0, "L1Loss": 1,
    "Wang": 0, "DortmundVital": 1, "Wang+DortmundVital": 2
})
_NUMERICAL_ENCODING_ADDS = MappingProxyType({
    "input_length": {5: 0, 10: 1, 20: 2},
    "eceoalldelta_input_length": {5: 0, 10: 1, 20: 2},
    "eceoalltheta_input_length": {5: 0, 10: 1, 20: 2},
    "eceoallalpha_input_length": {5: 0, 10: 1, 20: 2},
    "eceoallbeta_input_length": {5: 0, 10: 1, 20: 2},
    "eceoallgamma_input_length": {5: 0, 10: 1, 20: 2},
    "sfreq_multiple": {2: 0, 3: 1},
    "eceoalldelta_sfreq_multiple": {2: 0, 3: 1},
    "eceoalltheta_sfreq_multiple": {2: 0, 3: 1},
    "eceoallalpha_sfreq_multiple": {2: 0, 3: 1},
    "eceoallbeta_sfreq_multiple": {2: 0, 3: 1},
    "eceoallgamma_sfreq_multiple": {2: 0, 3: 1},
    "ocular_state": {"ec": 0, "eo": 1},
})


def main():
    # -------------
    # A few things to select
    # -------------
    studies = ("prediction_models", "pretraining", "simple_elecssl", "multivariable_elecssl")
    experiment_time = "2025-04-01_171150"

    selected_hps: Optional[Dict[str, Tuple[str, ...]]] = None

    num_trees = 8
    fanova_kwargs = {"n_trees": num_trees, "max_depth": 16}

    experiment_name = f"experiments_{experiment_time}"
    experiments_path = Path(get_results_dir() / experiment_name)

    # -------------
    # Make plots
    # -------------
    for study_name in studies:
        study_path = (experiments_path / study_name / f"{study_name}-study.db")
        study_storage = f"sqlite:///{study_path}"

        # Load optuna study object
        study = optuna.load_study(study_name=f"{study_name}-study", storage=study_storage)

        # Compute HP importance scores
        trials_df: pandas.DataFrame = study.trials_dataframe()

        # Restrict to unconditional HPs
        unconditional_hps = [col for col in trials_df.columns if col.startswith("params_")
                             and not trials_df[col].isnull().values.any()]
        trials_df = trials_df[["value"] + unconditional_hps]

        # Remove annoying prefix
        _prefix = "params_"
        col_mapping = {col: col[len(_prefix):] for col in trials_df.columns if col.startswith(_prefix)}
        trials_df.rename(columns=col_mapping, inplace=True)

        # Maybe use a selection only
        if selected_hps is not None:
            trials_df = trials_df[["value"] + list(selected_hps[study_name])]

        # Convert some of the categoricals to numerical encoding
        trials_df.replace(_NUMERICAL_ENCODING, inplace=True)
        for name, mapping in _NUMERICAL_ENCODING_ADDS.items():
            if name in trials_df.columns:
                trials_df[name].replace(mapping, inplace=True)

        # Drop nans
        trials_df.dropna(inplace=True)  # todo: drop them or say they are bad?
        trials_df.reset_index(drop=True, inplace=True)

        # Create the configuration space
        distributions = {param_name: param_dist for param_name, param_dist in study.trials[-1].distributions.items()
                         if param_name in trials_df.columns}
        config_space = ConfigSpace.ConfigurationSpace(_optuna_to_configspace(distributions))
        print(config_space)

        # Create object
        fanova_object = UpdatedFANOVA(X=trials_df.drop("value", axis="columns", inplace=False), Y=trials_df["value"],
                                      config_space=config_space, **fanova_kwargs
                                      )

        # todo: set cutoffs / percentile

        # Compute marginals
        marginal_importance: Dict[str, List[Any]] = {
            "HP": [], "Mean": [], "Std": [], **{f"Tree {i}": [] for i in range(num_trees)}
        }
        for hp_name in trials_df.columns:
            if hp_name == "value":
                continue

            summary = fanova_object.quantify_importance(dims=(hp_name,))[(hp_name,)]

            importance = summary["individual importance"]
            std = summary["individual std"]

            # Add to marginal importance
            marginal_importance["Mean"].append(importance)
            marginal_importance["Std"].append(std)

            # Add the rest of the info to marginal importance results
            marginal_importance["HP"].append(hp_name)

            for tree in range(num_trees):
                marginal_importance[f"Tree {tree}"].append(summary[f"Tree {tree}"])

        # Compute HP interaction effects
        hp_interaction_ranking = fanova_object.get_most_important_pairwise_marginals(n=None)
        pairwise_marginals = {"Rank": [], "HP1": [], "HP2": [], "Mean": [], "Std": [],
                              **{f"Tree {i}": [] for i in range(num_trees)}}
        for rank, ((hp_1, hp_2), (importance, std, all_trees)) in enumerate(hp_interaction_ranking.items()):
            # Add results. The HPs are ranked naturally
            pairwise_marginals["HP1"].append(hp_1)
            pairwise_marginals["HP2"].append(hp_2)
            pairwise_marginals["Rank"].append(rank)
            # pairwise_marginals["Percentile"].append(percentile)
            pairwise_marginals["Mean"].append(importance)
            pairwise_marginals["Std"].append(std)
            for tree, imp in all_trees.items():
                pairwise_marginals[tree].append(imp)

        # Make dataframe
        hp_pairwise_marginals_df = pandas.DataFrame(pairwise_marginals)
        hp_pairwise_marginals_df["HP"] = hp_pairwise_marginals_df["HP1"] + "\n+ " + hp_pairwise_marginals_df["HP2"]

        pyplot.figure()
        seaborn.barplot(hp_pairwise_marginals_df, y="Mean", x="HP")

    pyplot.show()


if __name__ == "__main__":
    main()
