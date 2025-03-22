"""
For suggesting hyperparameters using optuna
"""
from typing import Any, Dict

import yaml  # type: ignore[import-untyped]
from optuna.samplers import GridSampler, RandomSampler, TPESampler, CmaEsSampler, GPSampler, PartialFixedSampler, \
    NSGAIISampler, NSGAIIISampler, QMCSampler, BruteForceSampler

from elecssl.models.mts_modules.getter import get_mts_module_type
from elecssl.models.region_based_pooling.hyperparameter_sampling import generate_partition_sizes


def _get_num_time_steps(preprocessed_config_path, freq_band, suggested_preprocessing_steps):
    with open(preprocessed_config_path) as file:
        f_max = yaml.safe_load(file)["FrequencyBands"][freq_band][-1]

    return f_max * suggested_preprocessing_steps["sfreq_multiple"] * suggested_preprocessing_steps["input_length"]


def suggest_dl_architecture(name, trial, config, suggested_preprocessing_steps, preprocessed_config_path, freq_band):
    name_prefix = "" if name is None else f"{name}_"

    # Architecture
    model = trial.suggest_categorical(f"{name_prefix}architecture", choices=config.keys())

    # May also need the number of time steps
    model_config = config[model].copy()
    if "num_time_steps" in model_config and model_config["num_time_steps"] == "UNAVAILABLE":
        num_time_steps = _get_num_time_steps(
            suggested_preprocessing_steps=suggested_preprocessing_steps, freq_band=freq_band,
            preprocessed_config_path=preprocessed_config_path
        )
        model_config["num_time_steps"] = num_time_steps

    # Suggest hyperparameters of the DL model
    dl_name = f"{name}_{model}" if name is None else model  # This ensures that, e.g., number of filters is not 'shared'
    # across the architectures
    kwargs = get_mts_module_type(model).suggest_hyperparameters(name, dl_name, model_config)

    return {"model": model, "kwargs": kwargs}


def suggest_loss(name, trial, config):
    name_prefix = "" if name is None else f"{name}_"

    loss = trial.suggest_categorical(f"{name_prefix}loss", **config["loss"])

    # Sample re-weighting
    weighter = trial.suggest_categorical(f"{name_prefix}weighter", **config["weighter"])

    if weighter is None:
        return {"loss": loss, "loss_kwargs": {"reduction": "mean"}, "weighter": weighter, "weighter_kwargs": {}}
    else:
        weighter_kwargs = {}
        for param_name, (distribution, distribution_kwargs) in config["weighter_kwargs"].items():
            weighter_kwargs[param_name] = make_trial_suggestion(
                trial=trial, method=distribution, kwargs=distribution_kwargs, name=f"{name_prefix}{param_name}"
            )

        return {"loss": loss, "loss_kwargs": {"reduction": "none"}, "weighter": weighter,
                "weighter_kwargs": weighter_kwargs}


def _suggest_rbp(name, trial, config, normalisation, cmmn):
    num_montage_splits = trial.suggest_int(f"{name}_num_montage_splits", **config["num_montage_splits"])
    share_all_pooling_modules = trial.suggest_categorical(f"{name}_share_all_pooling_modules",
                                                          **config["share_all_pooling_modules"])
    if share_all_pooling_modules:
        num_pooling_modules = 1
    else:
        num_pooling_modules = max(1, round(
            num_montage_splits * trial.suggest_float(f"{name}_num_pooling_modules_percentage",
                                                     **config["num_pooling_modules_percentage"])
        ))

    # Generate number of montages for the pooling modules
    partitions = generate_partition_sizes(n=num_montage_splits, k=num_pooling_modules)

    # Generate all pooling module designs
    rbp_designs: Dict[str, Dict[str, Any]] = dict()
    for i, k in enumerate(partitions):
        rbp_name = f"RBPDesign{i}"
        rbp_designs[rbp_name] = dict()

        # CMMN
        rbp_designs[rbp_name]["cmmn_kwargs"] = cmmn["kwargs"]
        rbp_designs[rbp_name]["use_cmmn_layer"] = cmmn["use_cmmn_layer"]

        # Number of designs
        rbp_designs[rbp_name]["num_designs"] = config["num_designs"]  # Should be 1

        # Pooling type
        rbp_designs[rbp_name]["pooling_type"] = config["pooling_type"]  # Should be multi_cs

        # Pooling module
        _pooling_method = trial.suggest_categorical(f"{name}_pooling_method_{i}", config["PoolingMethods"].keys())
        rbp_designs[rbp_name]["pooling_methods"] = _pooling_method
        rbp_designs[rbp_name]["pooling_methods_kwargs"] = {}
        for param_name, (distribution, distribution_kwargs) in config["PoolingMethods"][_pooling_method].items():
            rbp_designs[rbp_name]["pooling_methods_kwargs"][param_name] = make_trial_suggestion(
                trial=trial, name=f"{name}_{param_name}_{i}", method=distribution, kwargs=distribution_kwargs
            )

        # Montage splits
        rbp_designs[rbp_name]["split_methods"] = []
        rbp_designs[rbp_name]["split_methods_kwargs"] = []
        for montage_split in range(k):
            # Name of montage split
            _name = trial.suggest_categorical(f"{name}_montage_split_{i}_{montage_split}",
                                              config["MontageSplits"].keys())
            rbp_designs[rbp_name]["split_methods"].append(_name)

            # Kwargs of montage split
            split_method_kwargs = dict()
            for param_name, (distribution, distribution_kwargs) in config["MontageSplits"][_name].items():
                split_method_kwargs[param_name] = make_trial_suggestion(
                    trial=trial, name=f"{name}_{param_name}_{i}_{montage_split}", method=distribution,
                    kwargs=distribution_kwargs
                )
            rbp_designs[rbp_name]["split_methods_kwargs"].append(split_method_kwargs)

    return {"name": "RegionBasedPooling", "kwargs": {"RBPDesigns": rbp_designs,
                                                     "normalise_region_representations": normalisation}}


def _suggest_interpolation(name, trial, config):
    method = trial.suggest_categorical(f"{name}_interpolation_method", **config["methods"])
    main_channel_system = trial.suggest_categorical(f"{name}_main_channel_system", **config["main_channel_system"])
    return {"name": "Interpolation", "kwargs": {"method": method, "main_channel_system": main_channel_system}}


def suggest_spatial_dimension_mismatch(name, trial, config, normalisation, cmmn):
    name_prefix = "" if name is None else f"{name}_"
    method = trial.suggest_categorical(f"{name_prefix}spatial_dimension_handling", **config["SpatialDimensionMismatch"])
    if method == "RegionBasedPooling":
        rbp_name = "rbp" if name is None else f"{name}_rbp"
        return _suggest_rbp(name=rbp_name, trial=trial, config=config["RegionBasedPooling"],
                            normalisation=normalisation, cmmn=cmmn)
    elif method == "Interpolation":
        interpolate_name = "interpolate" if name is None else f"{name}_interpolate"
        return _suggest_interpolation(name=interpolate_name, trial=trial, config=config["Interpolation"])
    else:
        raise ValueError(f"Unrecognised method for handling varied numbers of channels: {method}")


# -------------
# Main functions
# -------------
def make_trial_suggestion(trial, *, name, method, kwargs):
    """
    Function for making a suggestion using optuna

    Parameters
    ----------
    trial : optuna.Trial
    name : str
        name of the HP
    method : str
        The distribution to sample from
    kwargs : dict[str, typing.Any]
        The keyword arguments of the sampling distribution

    Examples
    --------
    >>> import optuna
    >>> my_study = optuna.create_study(direction="maximize")
    >>> my_trial = my_study.ask()
    >>> make_trial_suggestion(my_trial, name="CatVar", method="categorical", kwargs={"choices": ("a",)})
    'a'
    >>> _ = my_study.tell(my_trial, 0.5)
    >>> my_study.best_trial.params
    {'CatVar': 'a'}
    """
    if method == "categorical":
        func = trial.suggest_categorical
    elif method == "int":
        func = trial.suggest_int
    elif method == "float":
        func = trial.suggest_float
    elif method == "categorical_dict":
        suggested_key = trial.suggest_categorical(name, choices=tuple(kwargs.keys()))
        return kwargs[suggested_key]
    elif method == "not_a_hyperparameter":
        # Trial should not register it
        return kwargs
    else:
        raise ValueError(f"Sampling distribution of HP '{name}' not recognised: {method}")
    return func(name, **kwargs)


def get_optuna_sampler(sampler, **kwargs):
    """Function for getting a specified optuna sampler"""
    # All available samplers must be included here
    availables = (GridSampler, RandomSampler, TPESampler, CmaEsSampler, GPSampler, PartialFixedSampler, NSGAIISampler,
                  NSGAIIISampler, QMCSampler, BruteForceSampler)

    # Loop through and select the correct one
    for optuna_sampler in availables:
        if sampler == optuna_sampler.__name__:
            return optuna_sampler(**kwargs)

    # If no match, an error is raised
    raise ValueError(f"The sampler '{sampler}' was not recognised. Please select among the following: "
                     f"{tuple(s.__name__ for s in availables)}")
