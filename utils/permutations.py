import torch
from nflows.transforms.permutations import Permutation
import nflows.utils.typechecks as check

class BlockPermutation(Permutation):
    """Permutes just a block of lenght `block_size` at a time.
    NOTE: we are moving a number of elements equal to `block_size` to the right."""

    def __init__(self, features, block_size, dim=1):
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        if not check.is_positive_int(block_size):
            raise ValueError("Block size must be a positive integer.")

        super().__init__(
            torch.hstack(
                [
                    torch.arange(features)[block_size:],
                    torch.arange(features)[:block_size],
                ]
            )
        )
