import warnings

import numpy as np
import torch
from torch.nn.functional import softplus

from nflows.transforms import splines
from nflows.transforms.base import Transform
from nflows.transforms.nonlinearities import (
    PiecewiseCubicCDF,
    PiecewiseLinearCDF,
    PiecewiseQuadraticCDF,
    PiecewiseRationalQuadraticCDF,
)
from nflows.utils import torchutils


class CouplingTransformM(Transform):
    """A base class for coupling layers. Supports 2D inputs (NxD), as well as 4D inputs for
    images (NxCxHxW). For images the splitting is done on the channel dimension, using the
    provided 1D mask."""

    def __init__(self, mask, transform_net_create_fn, unconditional_transform=None, init_identity=True):
        """
        Constructor.
        Args:
            mask: a 1-dim tensor, tuple or list. It indexes inputs as follows:
                * If `mask[i] > 0`, `input[i]` will be transformed.
                * If `mask[i] <= 0`, `input[i]` will be passed unchanged.
        """
        mask = torch.as_tensor(mask)
        if mask.dim() != 1:
            raise ValueError("Mask must be a 1-dim tensor.")
        if mask.numel() <= 0:
            raise ValueError("Mask can't be empty.")

        super().__init__()
        self.features = len(mask)
        features_vector = torch.arange(self.features)

        self.min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE

        self.register_buffer(
            "identity_features", features_vector.masked_select(mask <= 0)
        )
        self.register_buffer(
            "transform_features", features_vector.masked_select(mask > 0)
        )

        assert self.num_identity_features + self.num_transform_features == self.features

        self.transform_net = transform_net_create_fn(
            self.num_identity_features,
            self.num_transform_features * self._transform_dim_multiplier(),
        )

        if init_identity:
          torch.nn.init.constant_(self.transform_net.final_layer.weight, 0.0)
          torch.nn.init.constant_(
              self.transform_net.final_layer.bias,
              np.log(np.exp(1 - self.min_derivative) - 1),
          )

        if unconditional_transform is None:
            self.unconditional_transform = None
        else:
            self.unconditional_transform = unconditional_transform(
                features=self.num_identity_features
            )

    @property
    def num_identity_features(self):
        return len(self.identity_features)

    @property
    def num_transform_features(self):
        return len(self.transform_features)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split, transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity = self.unconditional_transform(
                identity_split, context
            )
            logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features, ...] = identity_split
        outputs[:, self.transform_features, ...] = transform_split

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4]:
            raise ValueError("Inputs must be a 2D or a 4D tensor.")

        if inputs.shape[1] != self.features:
            raise ValueError(
                "Expected features = {}, got {}.".format(self.features, inputs.shape[1])
            )

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(
                identity_split, context
            )

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split, transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        return outputs, logabsdet

    def _transform_dim_multiplier(self):
        """Number of features to output for each transform dimension."""
        raise NotImplementedError()

    def _coupling_transform_forward(self, inputs, transform_params):
        """Forward pass of the coupling transform."""
        raise NotImplementedError()

    def _coupling_transform_inverse(self, inputs, transform_params):
        """Inverse of the coupling transform."""
        raise NotImplementedError()



class PiecewiseCouplingTransformM(CouplingTransformM):
    def _coupling_transform_forward(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=False)

    def _coupling_transform_inverse(self, inputs, transform_params):
        return self._coupling_transform(inputs, transform_params, inverse=True)

    def _coupling_transform(self, inputs, transform_params, inverse=False):
        if inputs.dim() == 4:
            b, c, h, w = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(
                0, 1, 3, 4, 2
            )
        elif inputs.dim() == 2:
            b, d = inputs.shape
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)

        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)

        return outputs, torchutils.sum_except_batch(logabsdet)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        raise NotImplementedError()