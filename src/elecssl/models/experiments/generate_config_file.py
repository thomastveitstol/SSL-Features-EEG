
def generate_config_file(config):
    # -----------------
    # Add details which can be copied directly from input config file
    # -----------------
    subjects_split_hyperparameters = config["SubjectSplit"]
    sub_groups_hyperparameters = config["SubGroups"]
    error_associations = config["PredictionErrorAssociations"]
    variables_metric = config["VariablesMetrics"]
    dl_architecture = config["DLArchitecture"]
    domain_discriminator = config["DomainDiscriminator"]
    varied_num_channels = config["Varied Numbers of Channels"]
    datasets = config["Datasets"]
    scalers = config["Scalers"]

    # -----------------
    # Add details which requires a little more handling
    # -----------------
    # Training config
    training = config["Training"]
    training["Loss"]["loss_kwargs"]["reduction"] = "mean" if training["Loss"]["weighter"] is None else "none"

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
