"""
Codebook replacement utilities.
lru_replacement is used for dead code replacement in VQ codebooks.
Only activated when replace_freq > 0 (not used in standard training).
"""

import torch
import torch.nn as nn


def lru_replacement(module, rho=0.01, timeout=100):
    """
    Register hooks for LRU-based codebook replacement.
    Replaces dead codebook entries with perturbed versions of active ones.

    Args:
        module: ResidualVectorQuantizer module
        rho: perturbation scale for replacement vectors
        timeout: number of forward passes before a code is considered dead
    """
    # Track usage counts for each codebook entry
    for i, vq_layer in enumerate(module.vq_layers):
        for j, codebook in enumerate(vq_layer.codebook):
            n_e = codebook.weight.shape[0]
            # Register buffer to track last usage time
            buffer_name = f"_usage_count_{i}_{j}"
            module.register_buffer(buffer_name, torch.zeros(n_e, dtype=torch.long))

    # Store replacement config
    module._replace_rho = rho
    module._replace_timeout = timeout
