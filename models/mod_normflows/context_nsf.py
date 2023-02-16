import torch
from torch import nn
import numpy as np

from normflows.flows.base import Flow
from normflows.flows.neural_spline.coupling import PiecewiseRationalQuadraticCoupling
from normflows.flows.neural_spline.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressive
from normflows.nets.resnet import ResidualNet
from normflows.utils.masks import create_alternating_binary_mask
# from normflows.utils.nn import PeriodicFeaturesElementwise
from normflows.utils.splines import DEFAULT_MIN_DERIVATIVE


class ContextCoupledRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [source](https://github.com/bayesiains/nsf)
    Added context to the coupling layer
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_context_channels=None,
        num_bins=8,
        tails="linear",
        tail_bound=3.0,
        activation=nn.ReLU,
        dropout_probability=0.0,
        reverse_mask=False,
        init_identity=True,
        batch_norm=False,
    ):
        """Constructor
        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_bins (int): Number of bins
          tails (str): Behaviour of the tails of the distribution, can be linear, circular for periodic distribution, or None for distribution on the compact interval
          tail_bound (float): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          reverse_mask (bool): Flag whether the reverse mask should be used
        """
        super().__init__()

        def transform_net_create_fn(in_features, out_features, num_context_channels=num_context_channels):
            return ResidualNet(
                in_features=in_features,
                out_features=out_features,
                context_features=num_context_channels,
                hidden_features=num_hidden_channels,
                num_blocks=num_blocks,
                activation=activation(),
                dropout_probability=dropout_probability,
                use_batch_norm=batch_norm,
            )

        self.prqct = PiecewiseRationalQuadraticCoupling(
            mask=create_alternating_binary_mask(num_input_channels, even=reverse_mask),
            transform_net_create_fn=transform_net_create_fn,
            num_bins=num_bins,
            tails=tails,
            tail_bound=tail_bound,
            # Setting True corresponds to equations (4), (5), (6) in the NSF paper:
            apply_unconditional_transform=True,
            # init_identity=init_identity,
        )

    def forward(self, z, context=None):
        z, log_det = self.prqct.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.prqct(z, context)
        return z, log_det.view(-1)


class ContextAutoregressiveRationalQuadraticSpline(Flow):
    """
    Neural spline flow coupling layer, wrapper for the implementation
    of Durkan et al., see [sources](https://github.com/bayesiains/nsf)
    """

    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_context_channels=None,
        num_bins=8,
        tail_bound=3,
        activation=nn.ReLU,
        dropout_probability=0.0,
        permute_mask=False,
        init_identity=True,
        batch_norm=False,
    ):
        """Constructor
        Args:
          num_input_channels (int): Flow dimension
          num_blocks (int): Number of residual blocks of the parameter NN
          num_hidden_channels (int): Number of hidden units of the NN
          num_bins (int): Number of bins
          tail_bound (int): Bound of the spline tails
          activation (torch module): Activation function
          dropout_probability (float): Dropout probability of the NN
          permute_mask (bool): Flag, permutes the mask of the NN
          init_identity (bool): Flag, initialize transform as identity
        """
        super().__init__()

        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=num_context_channels,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=batch_norm,
            init_identity=init_identity,
        )

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context)
        return z, log_det.view(-1)