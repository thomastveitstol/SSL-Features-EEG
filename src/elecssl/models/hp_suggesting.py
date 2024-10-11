"""
For suggesting hyperparameters using optuna
"""
from typing import Any, Dict

from optuna.samplers import GridSampler, RandomSampler, TPESampler, CmaEsSampler, GPSampler, PartialFixedSampler, \
    NSGAIISampler, NSGAIIISampler, QMCSampler, BruteForceSampler

from elecssl.models.mts_modules.getter import get_mts_module_type
from elecssl.models.region_based_pooling.hyperparameter_sampling import generate_partition_sizes


def _suggest_dl_architecture(name, trial, config):
    # Architecture
    model = trial.suggest_categorical(f"{name}_architecture", config.keys())

    # Suggest hyperparameters of the DL model
    kwargs = get_mts_module_type(model).suggest_hyperparameters(name, trial, config[model])

    return {"model": model, "kwargs": kwargs}


def _suggest_loss(name, trial, config):
    loss = trial.suggest_categorical(f"{name}_loss", **config["loss"])

    # Sample re-weighting
    weighter = trial.suggest_categorical(f"{name}_weighter", **config["weighter"])

    if weighter is None:
        return {"loss": loss, "loss_kwargs": {"reduction": "mean"}, "weighter": weighter}
    else:
        weighter_kwargs = {}
        for param_name, (distribution, distribution_kwargs) in config["weighter_kwargs"].items():
            weighter_kwargs[param_name] = make_trial_suggestion(
                trial=trial, method=distribution, kwargs=distribution_kwargs, name=f"{name}_{param_name}"
            )

        return {"loss": loss, "loss_kwargs": {"reduction": "none"}, "weighter": weighter,
                "weighter_kwargs": weighter_kwargs}


def _suggest_rbp(name, trial, config):
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
        for montage_split in range(k):
            # Name of montage split
            _name = trial.suggest_categorical(f"{name}_montage_split_{i}_{montage_split}",
                                              config["MontageSplits"].keys())
            rbp_designs[rbp_name]["split_methods"].append(_name)

            # Kwargs of montage split
            rbp_designs[rbp_name]["split_methods_kwargs"] = {}
            for param_name, (distribution, distribution_kwargs) in config["MontageSplits"][_name].items():
                rbp_designs[rbp_name]["split_methods_kwargs"][param_name] = make_trial_suggestion(
                    trial=trial, name=f"{name}_{param_name}_{i}_{montage_split}", method=distribution,
                    kwargs=distribution_kwargs
                )

    return rbp_designs


def _suggest_interpolation(name, trial, config):
    method = trial.suggest_categorical(f"{name}_interpolation_method", **config["methods"])
    main_channel_system = trial.suggest_categorical(f"{name}_main_channel_system", **config["main_channel_system"])
    return {"method": method, "main_channel_system": main_channel_system}


def _suggest_spatial_dimension_mismatch(name, trial, config):
    method = trial.suggest_categorical(f"{name}_spatial_dimension_handling", **config["SpatialDimensionMismatch"])
    if method == "RegionBasedPooling":
        return _suggest_rbp(name=name, trial=trial, config=config["RegionBasedPooling"])
    elif method == "Interpolation":
        return _suggest_interpolation(name=name, trial=trial, config=config["Interpolation"])
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
        suggested_key =  trial.suggest_categorical(name, choices=tuple(kwargs.keys()))
        return kwargs[suggested_key]
    else:
        raise ValueError(f"Sampling distribution of HP '{name}' not recognised: {method}")
    return func(name, **kwargs)


def suggest_hyperparameters(name, config, trial):
    """
    Function for suggesting HPs using optuna

    Parameters
    name : str
    config : dict[str, typing.Any]
    trial : optuna.Trial

    Returns
    -------
    dict[str, typing.Any]

    Examples
    --------
    >>> import optuna
    >>> my_study = optuna.create_study(direction="maximize")
    >>> my_trial = my_study.ask()
    >>> my_config = {"Preprocessing": {"autoreject": ("categorical", { "choices": (True,) })},
    ...              "Training": {"lr": ("float", {"low": 0.01, "high": 0.01})},
    ...              "DLArchitectures": {"InceptionNetwork": {"num_classes": 1, "cnn_units": {"low": 8, "high": 8},
    ...                                                       "depth": {"low": 3, "high": 3}}},
    ...              "Loss": {"loss": {"choices": ("L1Loss",)}, "weighter": {"choices": ("SamplePowerWeighter",)},
    ...                       "weighter_kwargs": {"weight_power": ("float", {"low": 0.2, "high": 0.2})}}}
    >>> suggest_hyperparameters(name="TRIAL", config=my_config, trial=my_trial)  # doctest: +NORMALIZE_WHITESPACE
    {'Preprocessing': {'autoreject': True}, 'Training': {'lr': 0.01},
     'DLArchitecture': {'model': 'InceptionNetwork', 'kwargs': {'cnn_units': 8, 'depth': 9, 'num_classes': 1}},
     'Loss': {'loss': 'L1Loss', 'loss_kwargs': {'reduction': 'none'}, 'weighter': 'SamplePowerWeighter',
               'weighter_kwargs': {'weight_power': 0.2}}}
    >>> _ = my_study.tell(my_trial, 0.5)
    >>> my_study.best_trial.params  # doctest: +NORMALIZE_WHITESPACE
    {'TRIAL_autoreject': True, 'TRIAL_lr': 0.01, 'TRIAL_architecture': 'InceptionNetwork', 'TRIAL_cnn_units': 8,
     'TRIAL_depth': 3.0, 'TRIAL_loss': 'L1Loss', 'TRIAL_weighter': 'SamplePowerWeighter', 'TRIAL_weight_power': 0.2}
    """
    suggested_hps: Dict[str, Any] = {"Preprocessing": {}, "Training": {}}

    # Preprocessing
    for param_name, (distribution, distribution_kwargs) in config["Preprocessing"].items():
        suggested_hps["Preprocessing"][param_name] = make_trial_suggestion(
            trial=trial, name=f"{name}_{param_name}", method=distribution, kwargs=distribution_kwargs
        )

    # Training
    for param_name, (distribution, distribution_kwargs) in config["Training"].items():
        suggested_hps["Training"][param_name] = make_trial_suggestion(
            trial=trial, name=f"{name}_{param_name}", method=distribution, kwargs=distribution_kwargs
        )

    # DL architecture
    suggested_hps["DLArchitecture"] = _suggest_dl_architecture(name=name, trial=trial, config=config["DLArchitectures"])

    # Loss
    suggested_hps["Loss"] = _suggest_loss(name=name, trial=trial, config=config["Loss"])

    # Handling varied numbers of electrodes
    suggested_hps["SpatialDimensionMismatch"] = _suggest_spatial_dimension_mismatch(name=name, trial=trial,
                                                                                    config=config)

    return suggested_hps


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
