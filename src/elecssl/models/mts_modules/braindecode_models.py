"""
Models provided by Braindecode are implemented here

Braindecode citation:
    Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M., Eggensperger, K., Tangermann, M.,
    Hutter, F., Burgard, W. and Ball, T. (2017), Deep learning with convolutional neural networks for EEG decoding and
    visualization. Hum. Brain Mapp., 38: 5391-5420. https://doi.org/10.1002/hbm.23730
"""
import torch
from braindecode.models import Deep4Net, ShallowFBCSPNet

from elecssl.models.mts_modules.mts_module_base import MTSModuleBase


class ShallowFBCSPNetMTS(MTSModuleBase):
    """
    The ShallowFBCSPNet architecture

    Paper:
        Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M., Eggensperger, K., Tangermann, M.,
        Hutter, F., Burgard, W. and Ball, T. (2017), Deep learning with convolutional neural networks for EEG decoding
        and visualization. Hum. Brain Mapp., 38: 5391-5420. https://doi.org/10.1002/hbm.23730

    Examples
    --------
    >>> _ = ShallowFBCSPNetMTS(4, 7, 200)

    Latent feature dimension does not depend on number of input channels

    >>> ShallowFBCSPNetMTS.get_latent_features_dim(19, 3, 100) == ShallowFBCSPNetMTS.get_latent_features_dim(64, 3, 100)
    True

    Latent features

    >>> ShallowFBCSPNetMTS(4, 7, 200).latent_features_dim
    280

    Number of time steps must be above 98

    >>> _ = ShallowFBCSPNetMTS(4, 7, 99)
    >>> _ = ShallowFBCSPNetMTS(4, 7, 98)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ValueError: During model prediction RuntimeError was thrown showing that at some layer ` Output size is too small`
    (see above in the stacktrace). This could be caused by providing too small `n_times`/`input_window_seconds`.
    Model may require longer chunks of signal in the input than (1, 4, 98).

    How the model looks like (the softmax/LogSoftmax activation function has been removed)

    >>> ShallowFBCSPNetMTS(4, 7, 200)  # doctest: +NORMALIZE_WHITESPACE
    ShallowFBCSPNetMTS(
      (_model): ShallowFBCSPNet(
        (ensuredims): Ensure4d()
        (dimshuffle): Rearrange('batch C T 1 -> batch 1 T C')
        (conv_time_spat): CombinedConv(
          (conv_time): Conv2d(1, 40, kernel_size=(25, 1), stride=(1, 1))
          (conv_spat): Conv2d(40, 40, kernel_size=(1, 4), stride=(1, 1), bias=False)
        )
        (bnorm): BatchNorm2d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_nonlin_exp): Expression(expression=square)
        (pool): AvgPool2d(kernel_size=(75, 1), stride=(15, 1), padding=0)
        (pool_nonlin_exp): Expression(expression=safe_log)
        (drop): Dropout(p=0.5, inplace=False)
        (final_layer): Sequential(
          (conv_classifier): Conv2d(40, 7, kernel_size=(7, 1), stride=(1, 1))
          (squeeze): Expression(expression=squeeze_final_output)
        )
      )
    )
    """

    expected_init_errors = (ValueError,)

    def __init__(self, in_channels, num_classes, num_time_steps, **kwargs):
        super().__init__()

        # ----------------
        # Initialise the model
        # ----------------
        self._model = ShallowFBCSPNet(n_chans=in_channels, n_outputs=num_classes, n_times=num_time_steps,
                                      final_conv_length="auto", add_log_softmax=False, **kwargs)

    def extract_latent_features(self, input_tensor):
        return self(input_tensor, return_features=True)

    def classify_latent_features(self, input_tensor):
        """
        Method for classifying the latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_batch, my_channels, my_time_steps = 10, 103, 600*3
        >>> my_model = ShallowFBCSPNetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> my_model.classify_latent_features(torch.rand(size=(10, 4560))).size()
        torch.Size([10, 3])

        Running (1) feature extraction and (2) classifying is the excact same as just running forward

        >>> my_model = ShallowFBCSPNetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> _ = my_model.eval()
        >>> my_input = torch.rand(size=(my_batch, my_channels, my_time_steps))
        >>> my_output_1 = my_model.classify_latent_features(my_model.extract_latent_features(my_input))
        >>> my_output_2 = my_model(my_input)
        >>> torch.equal(my_output_1, my_output_2)
        True
        """
        shape = (input_tensor.size()[0], self._model.final_layer.conv_classifier.in_channels,
                 self._model.final_conv_length, 1)
        return self._model.final_layer(torch.reshape(input_tensor, shape=shape))

    def forward(self, x, return_features=False):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        return_features : bool

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 600*3
        >>> my_model = ShallowFBCSPNetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps)), return_features=True).size()
        torch.Size([10, 4560])
        """
        # If predictions are to be made, just run forward method of the braindecode method
        if not return_features:
            return self._model(x)

        activations_name = "latent_features"
        activation = dict()

        # noinspection PyUnusedLocal
        def hook(model, inputs):
            if len(inputs) != 1:
                raise ValueError(f"Expected only one input, but received {len(inputs)}")
            activation[activations_name] = inputs[0].detach()

        self._model.final_layer.register_forward_pre_hook(hook)

        # Run forward method, but we are interested the latent features
        _ = self._model(x)
        latent_features = activation[activations_name]

        # Fix dimensions
        latent_features = torch.squeeze(latent_features, dim=-1)  # Removing redundant dimension
        latent_features = torch.reshape(latent_features, shape=(latent_features.size()[0], -1))
        return latent_features

    # ----------------
    # Hyperparameter sampling
    # ----------------
    @staticmethod
    def suggest_hyperparameters(name, trial, config):
        # Sample number of filters. Will be the same for temporal and spatial
        num_filters = trial.suggest_int(f"{name}_num_filters", **config["num_filters"])

        # Sample length of temporal filter
        filter_time_length = trial.suggest_int(f"{name}_filter_time_length", **config["filter_time_length"])

        # Sample length of temporal filter
        pool_time_stride = trial.suggest_int(f"{name}_pool_time_stride", **config["pool_time_stride"])

        # We set the ratio of length/stride to the same as in the original paper
        pool_time_length = 5 * pool_time_stride

        # Sample drop out
        drop_prob = trial.suggest_float(f"{name}_drop_prob", **config["drop_prob"])

        return {"n_filters_time":  num_filters,
                "n_filters_spat": num_filters,
                "filter_time_length": filter_time_length,
                "pool_time_stride": pool_time_stride,
                "pool_time_length": pool_time_length,
                "drop_prob": drop_prob,
                "num_classes": config["num_classes"],
                "num_time_steps": config["num_time_steps"]}

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self):
        # The latent features dimension is inferred from the dimension of their 'classifier_conv'
        return self._model.final_layer.conv_classifier.in_channels * self._model.final_conv_length


class Deep4NetMTS(MTSModuleBase):
    """
    The Deep4Net architecture

    Paper:
        Schirrmeister, R.T., Springenberg, J.T., Fiederer, L.D.J., Glasstetter, M., Eggensperger, K., Tangermann, M.,
        Hutter, F., Burgard, W. and Ball, T. (2017), Deep learning with convolutional neural networks for EEG decoding
        and visualization. Hum. Brain Mapp., 38: 5391-5420. https://doi.org/10.1002/hbm.23730

    Examples
    --------
    >>> _ = Deep4NetMTS(19, 3, 1000)

    Since padding on the conv layers was added, 160 time steps are allowed (the minimum is 89)

    Latent feature dimension does not depend on number of input channels

    >>> Deep4NetMTS.get_latent_features_dim(19, 3, 1000) == Deep4NetMTS.get_latent_features_dim(64, 3, 1000)
    True

    >>> _ = Deep4NetMTS(19, 3, 160)
    >>> _ = Deep4NetMTS(19, 3, 90)
    >>> _ = Deep4NetMTS(19, 3, 89, filter_time_length=10)  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    ZeroDivisionError: float division by zero

    How the model looks like

    >>> Deep4NetMTS(19, 3, 180)  # doctest: +NORMALIZE_WHITESPACE
    Deep4NetMTS(
      (_model): Deep4Net(
        (ensuredims): Ensure4d()
        (dimshuffle): Rearrange('batch C T 1 -> batch 1 T C')
        (conv_time_spat): CombinedConv(
          (conv_time): Conv2d(1, 25, kernel_size=(10, 1), stride=(1, 1))
          (conv_spat): Conv2d(25, 25, kernel_size=(1, 19), stride=(1, 1), bias=False)
        )
        (bnorm): BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv_nonlin): Expression(expression=elu)
        (pool): MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        (pool_nonlin): Expression(expression=identity)
        (drop_2): Dropout(p=0.5, inplace=False)
        (conv_2): Conv2d(25, 50, kernel_size=(10, 1), stride=(1, 1), padding=same, bias=False)
        (bnorm_2): BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlin_2): Expression(expression=elu)
        (pool_2): MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        (pool_nonlin_2): Expression(expression=identity)
        (drop_3): Dropout(p=0.5, inplace=False)
        (conv_3): Conv2d(50, 100, kernel_size=(10, 1), stride=(1, 1), padding=same, bias=False)
        (bnorm_3): BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlin_3): Expression(expression=elu)
        (pool_3): MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        (pool_nonlin_3): Expression(expression=identity)
        (drop_4): Dropout(p=0.5, inplace=False)
        (conv_4): Conv2d(100, 200, kernel_size=(10, 1), stride=(1, 1), padding=same, bias=False)
        (bnorm_4): BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlin_4): Expression(expression=elu)
        (pool_4): MaxPool2d(kernel_size=(3, 1), stride=(3, 1), padding=0, dilation=1, ceil_mode=False)
        (pool_nonlin_4): Expression(expression=identity)
        (final_layer): Sequential(
          (conv_classifier): Conv2d(200, 3, kernel_size=(2, 1), stride=(1, 1))
          (squeeze): Expression(expression=squeeze_final_output)
        )
      )
    )

    Does not work with 'split_last_layer' set to False

    >>> Deep4NetMTS(19, 3, 1000, split_first_layer=False)
    Traceback (most recent call last):
    ...
    AttributeError: 'Deep4Net' object has no attribute 'conv_time_spat'
    """

    def __init__(self, in_channels, num_classes, num_time_steps, **kwargs):
        super().__init__()

        # ----------------
        # Build the model
        # ----------------
        # Compute length of the final conv layer
        _final_conv_length = (num_time_steps - kwargs.get("filter_time_length", 10) + 1) // 3 // 3 // 3 // 3

        # Initialise from Braindecode
        model = Deep4Net(n_chans=in_channels, n_outputs=num_classes, n_times=num_time_steps,
                         final_conv_length=_final_conv_length, add_log_softmax=False, **kwargs)

        # Set padding. It was such a horror with the first one, so I just gave it up...
        model.conv_2.padding = "same"
        model.conv_3.padding = "same"
        model.conv_4.padding = "same"

        # Set attribute
        self._model = model

    def extract_latent_features(self, input_tensor):
        """
        Method for extracting latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_model = Deep4NetMTS(in_channels=4, num_classes=3, num_time_steps=1800,
        ...                        n_filters_4=206)
        >>> my_model.extract_latent_features(torch.rand(size=(10, 4, 1800))).size()
        torch.Size([10, 4532])
        """
        return self(input_tensor, return_features=True)

    def classify_latent_features(self, input_tensor):
        """
        Method for classifying the latent features

        Parameters
        ----------
        input_tensor : torch.Tensor

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> my_model = Deep4NetMTS(in_channels=4, num_classes=8, num_time_steps=1800, n_filters_4=206)
        >>> my_model.classify_latent_features(torch.rand(size=(10, 4532))).size()
        torch.Size([10, 8])

        Running (1) feature extraction and (2) classifying is the excact same as just running forward

        >>> my_model = Deep4NetMTS(in_channels=19, num_classes=8, num_time_steps=1500)
        >>> _ = my_model.eval()
        >>> my_input = torch.rand(size=(10, 19, 1500))
        >>> my_output_1 = my_model.classify_latent_features(my_model.extract_latent_features(my_input))
        >>> my_output_2 = my_model(my_input)
        >>> torch.equal(my_output_1, my_output_2)
        True
        """
        shape = (input_tensor.size()[0], self._model.n_filters_4, self._model.final_conv_length, 1)
        return self._model.final_layer(torch.reshape(input_tensor, shape=shape))

    def forward(self, x, return_features=False):
        """
        Forward method

        Parameters
        ----------
        x : torch.Tensor
        return_features : bool

        Returns
        -------
        torch.Tensor

        Examples
        --------
        >>> import torch
        >>> my_batch, my_channels, my_time_steps = 10, 103, 1800
        >>> my_model = Deep4NetMTS(in_channels=my_channels, num_classes=3, num_time_steps=my_time_steps,
        ...                        n_filters_4=206)
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps))).size()
        torch.Size([10, 3])
        >>> my_model(torch.rand(size=(my_batch, my_channels, my_time_steps)), return_features=True).size()
        torch.Size([10, 4532])

        New example t(his one didn't work in a previous implementation)

        >>> my_model = Deep4NetMTS(filter_time_length=11, filter_length_2=11, filter_length_3=11, filter_length_4=11,
        ...                        n_filters_2=56, n_filters_3=112, n_filters_4=224, n_filters_spat=28,
        ...                        n_filters_time=28, num_time_steps=900, drop_prob=0.075167, num_classes=1,
        ...                        in_channels=19)
        >>> my_model(torch.rand(16, 19, 900)).size()
        torch.Size([16, 1])
        """
        # If predictions are to be made, just run forward method of the braindecode method
        if not return_features:
            return self._model(x)

        activations_name = "latent_features"
        activation = dict()

        # noinspection PyUnusedLocal
        def hook(model, inputs):
            if len(inputs) != 1:
                raise ValueError(f"Expected only one input, but received {len(inputs)}")
            activation[activations_name] = inputs[0].detach()

        self._model.final_layer.register_forward_pre_hook(hook)

        # Run forward method, but we are interested the latent features
        _ = self._model(x)
        latent_features = activation[activations_name]

        # Fix dimensions. Currently, shape=(batch, n_filters_4, final_conv_length, 1)
        latent_features = torch.squeeze(latent_features, dim=-1)  # Removing redundant dimension
        latent_features = torch.reshape(latent_features, shape=(latent_features.size()[0], -1))
        return latent_features

    # ----------------
    # Hyperparameter sampling
    # ----------------
    @staticmethod
    def suggest_hyperparameters(name, trial, config):
        """The ratio between the number of filters will be maintained as in the original work"""
        # Get the number of filters for the first conv block
        num_first_filters = trial.suggest_int(f"{name}_num_first_filters", **config["num_first_filters"])

        num_filters_hyperparameters = {"n_filters_time": num_first_filters,
                                       "n_filters_spat": num_first_filters,
                                       "n_filters_2": 2 * num_first_filters,
                                       "n_filters_3": 4 * num_first_filters,
                                       "n_filters_4": 8 * num_first_filters}

        # Get the filter lengths
        filter_length = trial.suggest_int(f"{name}_filter_length", **config["filter_length"])

        # Compute the length of the filters for the conv blocks
        filter_lengths_hyperparameters = {"filter_time_length": filter_length,
                                          "filter_length_2": filter_length,
                                          "filter_length_3": filter_length,
                                          "filter_length_4": filter_length}
        # Get the drop out
        drop_prob = trial.suggest_float(f"{name}_drop_prob", **config["drop_prob"])

        return {**num_filters_hyperparameters, **filter_lengths_hyperparameters, "drop_prob": drop_prob,
                "num_time_steps": config["num_time_steps"], "num_classes": config["num_classes"]}

    # ----------------
    # Properties
    # ----------------
    @property
    def latent_features_dim(self):
        # The latent features dimension is inferred from the dimension of their 'classifier_conv'
        return self._model.n_filters_4 * self._model.final_conv_length
