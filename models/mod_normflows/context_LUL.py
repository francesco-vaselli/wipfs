# a custom context flow based on the LULinearPermute flow from normflows
from normflows.flows.mixing import _LULinear, _RandomPermutation
from normflows.flows.base import Flow


class ContextLULinearPermute(Flow):
    """
    Fixed permutation combined with a linear transformation parametrized
    using the LU decomposition, used in https://arxiv.org/abs/1906.04032
    Added context keyword to forward and inverse methods to allow use in context flows
    """

    def __init__(self, num_channels, identity_init=True):
        """Constructor
        Args:
          num_channels: Number of dimensions of the data
          identity_init: Flag, whether to initialize linear transform as identity matrix
        """
        # Initialize
        super().__init__()

        # Define modules
        self.permutation = _RandomPermutation(num_channels)
        self.linear = _LULinear(num_channels, identity_init=identity_init)

    def forward(self, z, context):
        z, log_det = self.linear.inverse(z)
        z, _ = self.permutation.inverse(z)
        return z, log_det.view(-1)

    def inverse(self, z, context):
        z, _ = self.permutation(z)
        z, log_det = self.linear(z)
        return z, log_det.view(-1)
