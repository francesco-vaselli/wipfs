"""Basic definitions for the distributions module, modded for allowing forward pass."""

import torch
from torch import nn

from nflows.utils import torchutils
import nflows.utils.typechecks as check

from inspect import signature

class NoMeanException(Exception):
    """Exception to be thrown when a mean function doesn't exist."""

    pass


class DistributionM(nn.Module):
    """modded Base class for all distribution objects.
    allows for forward method to be called"""

    def forward(self, inputs, context=None):
        # raise RuntimeError("Forward method cannot be called for a Distribution object.")
        return self.log_prob(inputs, context)

    def log_prob(self, inputs, context=None):
        """Calculate log probability under the distribution.
        Args:
            inputs: Tensor, input variables.
            context: Tensor or None, conditioning variables. If a Tensor, it must have the same
                number or rows as the inputs. If None, the context is ignored.
        Returns:
            A Tensor of shape [input_size], the log probability of the inputs given the context.
        """
        inputs = torch.as_tensor(inputs)
        if context is not None:
            context = torch.as_tensor(context)
            if inputs.shape[0] != context.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of context items."
                )
        return self._log_prob(inputs, context)

    def _log_prob(self, inputs, context):
        raise NotImplementedError()

    def sample(self, num_samples, context=None, batch_size=None):
        """Generates samples from the distribution. Samples can be generated in batches.
        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored. 
                     Should have shape [context_size, ...], where ... represents a (context) feature 
                     vector of arbitrary shape. This will generate num_samples for each context item 
                     provided. The overall shape of the samples will then be 
                     [context_size, num_samples, ...].
            batch_size: int or None, number of samples per batch. If None, all samples are generated
                in one batch.
        Returns:
            A Tensor containing the samples, with shape [num_samples, ...] if context is None, or
            [context_size, num_samples, ...] if context is given, where ... represents a feature
            vector of arbitrary shape.
        """
        if not check.is_positive_int(num_samples):
            raise TypeError("Number of samples must be a positive integer.")

        if context is not None:
            context = torch.as_tensor(context)

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            if not check.is_positive_int(batch_size):
                raise TypeError("Batch size must be a positive integer.")

            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def _sample(self, num_samples, context):
        raise NotImplementedError()

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the distribution together with their log probability.
        Args:
            num_samples: int, number of samples to generate.
            context: Tensor or None, conditioning variables. If None, the context is ignored. 
                     Should have shape [context_size, ...], where ... represents a (context) feature 
                     vector of arbitrary shape. This will generate num_samples for each context item 
                     provided. The overall shape of the samples will then be 
                     [context_size, num_samples, ...].
        Returns:
            A tuple of:
                * A Tensor containing the samples, with shape [num_samples, ...] if context is None,
                  or [context_size, num_samples, ...] if context is given, where ... represents a 
                  feature vector of arbitrary shape.
                * A Tensor containing the log probabilities of the samples, with shape
                  [num_samples, features if context is None, or [context_size, num_samples, ...] if
                  context is given.
        """
        samples = self.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to call log_prob.
            samples = torchutils.merge_leading_dims(samples, num_dims=2)
            context = torchutils.repeat_rows(context, num_reps=num_samples)
            assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            log_prob = torchutils.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob

    def mean(self, context=None):
        if context is not None:
            context = torch.as_tensor(context)
        return self._mean(context)

    def _mean(self, context):
        raise NoMeanException()


class FlowM(DistributionM):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution, embedding_net=None):
        """Constructor.
        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
            embedding_net: A `nn.Module` which has trainable parameters to encode the
                context (condition). It is trained jointly with the flow.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        distribution_signature = signature(self._distribution.log_prob)
        distribution_arguments =  distribution_signature.parameters.keys()
        self._context_used_in_base = 'context' in distribution_arguments
        if embedding_net is not None:
            assert isinstance(embedding_net, torch.nn.Module), (
                "embedding_net is not a nn.Module. "
                "If you want to use hard-coded summary features, "
                "please simply pass the encoded features and pass "
                "embedding_net=None"
            )
            self._embedding_net = embedding_net
        else:
            self._embedding_net = torch.nn.Identity()

    def _log_prob(self, inputs, context):
        embedded_context = self._embedding_net(context)
        noise, logabsdet = self._transform(inputs, context=embedded_context)
        if self._context_used_in_base:
            log_prob = self._distribution.log_prob(noise, context=embedded_context)
        else:
            log_prob = self._distribution.log_prob(noise)
        return log_prob,  logabsdet # changed to get the separate contributions

    def _sample(self, num_samples, context):
        embedded_context = self._embedding_net(context)
        if self._context_used_in_base:
            noise = self._distribution.sample(num_samples, context=embedded_context)
        else:
            repeat_noise = self._distribution.sample(num_samples*embedded_context.shape[0])
            noise = torch.reshape(
                    repeat_noise,
                    (embedded_context.shape[0], -1, repeat_noise.shape[1])
                    )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, _ = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.
        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        embedded_context = self._embedding_net(context)
        if self._context_used_in_base:
            noise, log_prob = self._distribution.sample_and_log_prob(
                num_samples, context=embedded_context
            )
        else:
            noise, log_prob = self._distribution.sample_and_log_prob(
                num_samples
            )

        if embedded_context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = torchutils.merge_leading_dims(noise, num_dims=2)
            embedded_context = torchutils.repeat_rows(
                embedded_context, num_reps=num_samples
            )

        samples, logabsdet = self._transform.inverse(noise, context=embedded_context)

        if embedded_context is not None:
            # Split the context dimension from sample dimension.
            samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = torchutils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None):
        """Transforms given data into noise. Useful for goodness-of-fit checking.
        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.
        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, _ = self._transform(inputs, context=self._embedding_net(context))
        return noise
