from elecssl.models.domain_adaptation.domain_discriminators.getter import get_domain_discriminator_type
from elecssl.models.mts_modules.getter import get_mts_module_type


def generate_config_file(config):
    # -----------------
    # Add details which can be copied directly from input config file
    # -----------------
    subjects_split_hyperparameters = config["SubjectSplit"]
    sub_groups_hyperparameters = config["SubGroups"]
    error_associations = config["PredictionErrorAssociations"]
    variables_metric = config["VariablesMetrics"]
    dl_architecture = config["DLArchitecture"]
    varied_num_channels = config["Varied Numbers of Channels"]
    datasets = config["Datasets"]
    scalers = config["Scalers"]

    # -----------------
    # Add details which requires a little more handling
    # -----------------
    # Training config
    training = config["Training"]
    training["Loss"]["loss_kwargs"]["reduction"] = "mean" if training["Loss"]["weighter"] is None else "none"

    # Domain discriminator
    domain_discriminator = config["DomainDiscriminator"].copy()
    if domain_discriminator["discriminator"] is not None:
        # Need to get and add the input dimension from the DL model
        input_dimension = get_mts_module_type(dl_architecture["model"]).get_latent_features_dim(
            in_channels=19, **dl_architecture["kwargs"]) # this will only work if the number of features is independent
        # of number of input channels, otherwise an error message will be raised during the experiment. Doctests have
        # shown that number of input channels does not affect latent feature dimension

        domain_discriminator["discriminator"]["kwargs"]["kwargs"]["in_features"] = input_dimension

        # With the current implementation, the architecture is sampled here (not directly in the yaml file)
        domain_discriminator["discriminator"]["kwargs"] = get_domain_discriminator_type(
            domain_discriminator["discriminator"]["name"]).sample_hyperparameters(
            method=domain_discriminator["discriminator"]["kwargs"]["method"],
            **domain_discriminator["discriminator"]["kwargs"]["kwargs"]
        )
    else:
        domain_discriminator = None

    return {
        "SubjectSplit": subjects_split_hyperparameters,
        "SubGroups": sub_groups_hyperparameters,
        "Scalers": scalers,
        "VariablesMetrics": variables_metric,
        "PredictionErrorAssociations": error_associations,
        "Datasets": datasets,
        "Varied Numbers of Channels": varied_num_channels,
        "DomainDiscriminator": domain_discriminator,
        "DLArchitecture": dl_architecture,
        "Training": training
    }
