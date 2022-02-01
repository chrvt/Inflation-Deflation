# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 11:16:30 2021
BNAF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math
from .made import MADE
# import splines import rational_quadratic_spline
# --------------------
# Model components
# --------------------

class Transform(nn.Module):
    """Base class for all transform objects."""

    def forward(self, inputs, context=None, full_jacobian=False):
        raise NotImplementedError()

    def inverse(self, inputs, context=None, full_jacobian=False):
        raise InverseNotAvailable()
        
        
class CompositeTransform(Transform):
    """Composes several transforms into one, in the order they are given."""

    def __init__(self, transforms):
        """Constructor.

        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs, context, full_jacobian=False):
        batch_size = inputs.shape[0]
        outputs = inputs

        if full_jacobian:
            total_jacobian = None
            for func in funcs:
                inputs = outputs
                outputs, jacobian = func(inputs, context, full_jacobian=True)

                # # Cross-check for debugging
                # _, logabsdet = func(inputs, context, full_jacobian=False)
                # _, logabsdet_from_jacobian = torch.slogdet(jacobian)
                # logger.debug("Transformation %s has Jacobian\n%s\nwith log abs det %s (ground truth %s)", type(func).__name__, jacobian.detach().numpy()[0], logabsdet_from_jacobian[0].item(), logabsdet[0].item())

                # timer.timer(start="Jacobian multiplication")
                total_jacobian = jacobian if total_jacobian is None else torch.bmm(jacobian, total_jacobian)
                # timer.timer(stop="Jacobian multiplication")

            # logger.debug("Composite Jacobians \n %s", total_jacobian[0])

            return outputs, total_jacobian

        else:
            total_logabsdet = torch.zeros(batch_size)
            for func in funcs:
                outputs, logabsdet = func(outputs, context)
                total_logabsdet += logabsdet
            return outputs, total_logabsdet

    def forward(self, inputs, context=None, full_jacobian=False):
        funcs = self._transforms
        return self._cascade(inputs, funcs, context, full_jacobian)

    def inverse(self, inputs, context=None, full_jacobian=False):
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs, context, full_jacobian)
    
    
class AutoregressiveTransform(Transform):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    NOTE: Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """

    def __init__(self, autoregressive_net):
        super(AutoregressiveTransform, self).__init__()
        self.autoregressive_net = autoregressive_net

    def forward(self, inputs, context=None, full_jacobian=False):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params, full_jacobian=full_jacobian)
        return outputs, logabsdet

    def inverse(self, inputs, context=None, full_jacobian=False):
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(inputs, autoregressive_params, full_jacobian=full_jacobian)
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params, full_jacobian=False):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params, full_jacobian=False):
        raise NotImplementedError()

        
        
class MaskedAffineAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):

        # self.mu = torch.zeros(features)
        # self.sigma = torch.ones(features)
        
        self.features = features
        made = MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )
        super(MaskedAffineAutoregressiveTransform, self).__init__(made)
        
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(features))
        self.register_buffer('base_dist_var', torch.ones(features))

    def _output_dim_multiplier(self):
        return 2

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var) #D.Normal(self.mu, self.sigma)

    def _elementwise_forward(self, inputs, autoregressive_params, full_jacobian=False):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 2.0) + 1e-3
        log_scale = torch.log(scale)
        outputs = scale * inputs + shift
        if full_jacobian:
            raise NotImplementedError
        else:
            logabsdet = log_scale # various.sum_except_batch(log_scale, num_batch_dims=1)
            return outputs, logabsdet

    def _elementwise_inverse(self, inputs, autoregressive_params, full_jacobian=False):
        unconstrained_scale, shift = self._unconstrained_scale_and_shift(autoregressive_params)
        scale = torch.sigmoid(unconstrained_scale + 2.0) + 1e-3
        log_scale = torch.log(scale)
        outputs = (inputs - shift) / scale
        if full_jacobian:
            raise NotImplementedError
        else:
            logabsdet = - log_scale #-various.sum_except_batch(log_scale, num_batch_dims=1)
            return outputs, logabsdet

    def _unconstrained_scale_and_shift(self, autoregressive_params):
        # split_idx = autoregressive_params.size(1) // 2
        # unconstrained_scale = autoregressive_params[..., :split_idx]
        # shift = autoregressive_params[..., split_idx:]
        # return unconstrained_scale, shift
        autoregressive_params = autoregressive_params.view(-1, self.features, self._output_dim_multiplier())
        unconstrained_scale = autoregressive_params[..., 0]
        shift = autoregressive_params[..., 1]
        return unconstrained_scale, shift
    
class MaskedPiecewiseRationalQuadraticAutoregressiveTransform(AutoregressiveTransform):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails
        self.tail_bound = tail_bound

        autoregressive_net = MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

        super().__init__(autoregressive_net)

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails is None:
            return self.num_bins * 3 + 1
        else:
            raise ValueError

    def _elementwise(self, inputs, autoregressive_params, inverse=False, full_jacobian=False):

        if full_jacobian:
            raise NotImplementedError

        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(batch_size, features, self._output_dim_multiplier())

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, various.sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params, full_jacobian=False):
        return self._elementwise(inputs, autoregressive_params, full_jacobian=full_jacobian)

    def _elementwise_inverse(self, inputs, autoregressive_params, full_jacobian=False):
        return self._elementwise(inputs, autoregressive_params, inverse=True, full_jacobian=full_jacobian)