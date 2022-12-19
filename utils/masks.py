from nflows.utils import typechecks as check
import numpy as np
import torch


def create_block_binary_mask(features, block_size):
    """
    Creates a binary mask of a given dimension which splits its masking into blocks of a given size.
    :param features: Dimension of mask.
    :param block_size: Size of blocks to split mask into.
    :return: Binary mask split into blocks of type torch.Tensor
    """
    if not check.is_positive_int(features):
        raise ValueError("Number of features must be a positive integer.")
    if not check.is_positive_int(block_size):
        raise ValueError("Block size must be a positive integer.")

    mask = torch.zeros(features).byte()
    mask[:block_size] += 1
    return mask


def create_identity_mask(features):
    """
    Leaves input unchanged.
    """
    if not check.is_positive_int(features):
        raise ValueError("Number of features must be a positive integer.")

    mask = torch.ones(features).byte()
    return mask
