import torch
import numpy as np
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


class IdentityPermutation(Permutation):
    """Leaves input unchanged."""

    def __init__(self, features, dim=1):
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")

        super().__init__(
            torch.arange(features),
        )


class TripletPermutation(Permutation):
    """Permutes the input in triplets.
    The first three elements are shuffled between them and so forth
    All the triplets are permuted in the same way"""

    def __init__(self, features, dim=1):
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        if features % 3 != 0:
            raise ValueError("Number of features must be a multiple of 3.")

        triplet_perm = torch.arange(3)[torch.randperm(3)]
        new_idx = torch.tensor(
            np.array(
                [n * 3 + triplet_perm for n in range(0, int(features / 3))]
            ).flatten()
        )
        super().__init__(
            torch.arange(features)[new_idx],
        )
