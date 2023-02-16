import torch
import torch.nn as nn
import numpy as np

from normflows import utils


def is_int(x):
    return isinstance(x, int)


def is_positive_int(x):
    return is_int(x) and x > 0


def split_leading_dim(x, shape):
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


class ContextNormalizingFlow(nn.Module):
    """
    Normalizing Flow model to approximate target distribution based on context
    NOTE: atm only forwad_kld and sample are implemented with context

    """

    def __init__(self, q0, flows, p=None):
        """Constructor
        Args:
          q0: Base distribution
          flows: List of flows
          p: Target distribution
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward(self, z):
        """Transforms latent variable z to the flow variable x
        Args:
          z: Batch in the latent space
        Returns:
          Batch in the space of the target distribution
        """
        for flow in self.flows:
            z, _ = flow(z)
        return z

    def forward_and_log_det(self, z):
        """Transforms latent variable z to the flow variable x and
        computes log determinant of the Jacobian
        Args:
          z: Batch in the latent space
        Returns:
          Batch in the space of the target distribution,
          log determinant of the Jacobian
        """
        log_det = torch.zeros(len(z), device=z.device)
        for flow in self.flows:
            z, log_d = flow(z)
            log_det -= log_d
        return z, log_det

    def inverse(self, x):
        """Transforms flow variable x to the latent variable z
        Args:
          x: Batch in the space of the target distribution
        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def inverse_and_log_det(self, x):
        """Transforms flow variable x to the latent variable z and
        computes log determinant of the Jacobian
        Args:
          x: Batch in the space of the target distribution
        Returns:
          Batch in the latent space, log determinant of the
          Jacobian
        """
        log_det = torch.zeros(len(x), device=x.device)
        for i in range(len(self.flows) - 1, -1, -1):
            x, log_d = self.flows[i].inverse(x)
            log_det += log_d
        return x, log_det

    def forward_kld(self, x, context=None):
        """
        Estimates forward KL divergence, see arXiv 1912.02762
        :param x: Batch sampled from target distribution
        :param context: Batch of context variables
        :return: Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, context=context)
            log_q += log_det
        # log_q += self.q0.log_prob(z)
        log_p = self.q0.log_prob(z)
        # return -torch.mean(log_q)
        return -torch.mean(log_p), -torch.mean(log_q)

    def reverse_kld(self, num_samples=1, beta=1.0, score_fn=True):
        """Estimates reverse KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)
        Args:
          num_samples: Number of samples to draw from base distribution
          beta: Annealing parameter, see [arXiv 1505.05770](https://arxiv.org/abs/1505.05770)
          score_fn: Flag whether to include score function in gradient, see [arXiv 1703.09194](https://arxiv.org/abs/1703.09194)
        Returns:
          Estimate of the reverse KL divergence averaged over latent samples
        """
        z, log_q_ = self.q0(num_samples)
        log_q = torch.zeros_like(log_q_)
        log_q += log_q_
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        if not score_fn:
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
        log_p = self.p.log_prob(z)
        return torch.mean(log_q) - beta * torch.mean(log_p)

    def reverse_alpha_div(self, num_samples=1, alpha=1, dreg=False):
        """Alpha divergence when sampling from q
        Args:
          num_samples: Number of samples to draw
          dreg: Flag whether to use Double Reparametrized Gradient estimator, see [arXiv 1810.04152](https://arxiv.org/abs/1810.04152)
        Returns:
          Alpha divergence
        """
        z, log_q = self.q0(num_samples)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.p.log_prob(z)
        if dreg:
            w_const = torch.exp(log_p - log_q).detach()
            z_ = z
            log_q = torch.zeros(len(z_), device=z_.device)
            utils.set_requires_grad(self, False)
            for i in range(len(self.flows) - 1, -1, -1):
                z_, log_det = self.flows[i].inverse(z_)
                log_q += log_det
            log_q += self.q0.log_prob(z_)
            utils.set_requires_grad(self, True)
            w = torch.exp(log_p - log_q)
            w_alpha = w_const**alpha
            w_alpha = w_alpha / torch.mean(w_alpha)
            weights = (1 - alpha) * w_alpha + alpha * w_alpha**2
            loss = -alpha * torch.mean(weights * torch.log(w))
        else:
            loss = np.sign(alpha - 1) * torch.logsumexp(alpha * (log_p - log_q), 0)
        return loss

    def sample(self, num_samples=1, context=None):
        """
        Samples from flow-based approximate distribution
        :param num_samples: Number of samples to draw
        :param context: Context for conditional flows (accepts batches and will output one sample per batch element)
        :return: Samples, log probability
        """

        if context is not None:
            repeat_noise, log_q = self.q0(num_samples * context.shape[0])
            z = torch.reshape(repeat_noise, (context.shape[0], -1, repeat_noise.shape[1]))
            # Merge the context dimension with sample dimension in order to apply the transform.
            z = merge_leading_dims(z, num_dims=2)
            context = repeat_rows(context, num_reps=num_samples)
            for flow in self.flows:
                z, log_det = flow(z, context=context)
                log_q -= log_det

            # Split the context dimension from sample dimension.
            z = split_leading_dim(z, shape=[-1, num_samples])

        else:
            z, log_q = self.q0(num_samples)

            for flow in self.flows:
                z, log_det = flow(z)
                log_q -= log_det

        return z, log_q

    def log_prob(self, x):
        """Get log probability for batch
        Args:
          x: Batch
        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z)
        return log_q

    def save(self, path):
        """Save state dict of model
        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict
        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))
