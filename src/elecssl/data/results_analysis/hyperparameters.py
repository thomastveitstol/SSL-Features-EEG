from ConfigSpace import CategoricalHyperparameter, UniformFloatHyperparameter


# --------------
# Individual HPs
# --------------
def to_hyperparameter(*, name, method, **kwargs):
    if method == "categorical":
        return CategoricalHyperparameter(name=name, **kwargs)
    elif method == "float":
        return UniformFloatHyperparameter(name=name, lower=kwargs["low"], upper=kwargs["high"], log=False)
    raise ValueError(f"Sampling distribution of HP '{name}' not recognised: {method}")


"""def get_normalisation_hpc(config):
    if config["Varied Numbers of Channels"]["name"] == "Interpolation":
        return config["DL Architecture"]["normalise"]
    elif config["Varied Numbers of Channels"]["name"] == "RegionBasedPooling":
        return config["Varied Numbers of Channels"]["kwargs"]["normalise_region_representations"]
    else:
        raise ValueError

def get_normalisation_hpd(config):
    return CategoricalHyperparameter(name="Normalisation", **config["normalisation"])

def get_dl_architecture_hpd(config):
    return CategoricalHyperparameter(name="DL architecture", choices=tuple(config["DLArchitectures"]))



CHP = CategoricalHyperparameter
OHP = OrdinalHyperparameter()
UFHP = UniformFloatHyperparameter()
HYPERPARAMETERS = MappingProxyType({
    "Normalisation": HP(hpc=_get_normalisation_hpc, hpd=_get_normalisation_hpd),
    "DL architecture": HP(("DLArchitecture", "model"), hpd=_get_dl_architecture_hpd),
    "Learning rate": HP,
    r"$\beta_1$": HP,
    r"$\beta_2$": HP
})"""
