# streetsense/selfsup/contrastive.py
# -----------------------------------------------------------
# Simple SimCLR-style utilities for self-supervised training.
# - nt_xent: temperature-scaled cross-entropy on cosine sims
# - augment_audio / augment_imu: lightweight sensor augmentations
#
# This implementation keeps a full [2N, 2N] similarity matrix and
# masks the diagonal with a large negative number so positives stay
# at stable indices. This avoids the out-of-bounds issue.
# -----------------------------------------------------------

from __future__ import annotations
import torch
import torch.nn.functional as F


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    Compute the NT-Xent (InfoNCE / SimCLR) loss between two batches of embeddings.

    Args:
        z1: [N, D] embeddings for view 1
        z2: [N, D] embeddings for view 2
        temperature: temperature for softmax scaling

    Returns:
        Scalar torch.Tensor loss.
    """
    assert z1.dim() == 2 and z2.dim() == 2, "z1/z2 must be [N, D]"
    assert z1.size(0) == z2.size(0) and z1.size(1) == z2.size(1), "z1/z2 must match"

    # Normalize just in case the encoders didn't (safe no-op if already normalized).
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    z = torch.cat([z1, z2], dim=0)             # [2N, D]
    sim = torch.matmul(z, z.t())               # cosine since rows are normalized -> [2N, 2N]

    N = z1.size(0)

    # Mask self-similarity on the diagonal so the model cannot trivially match with itself.
    diag_mask = torch.eye(2 * N, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag_mask, -1e9)     # keep shape [2N, 2N]

    # Positive indices: for i in [0..N-1], positive is i+N; for i in [N..2N-1], positive is i-N.
    labels = torch.cat([
        torch.arange(N, 2 * N, device=z.device),
        torch.arange(0, N, device=z.device)
    ], dim=0)                                   # [2N]

    logits = sim / temperature                  # [2N, 2N]
    loss = F.cross_entropy(logits, labels)
    return loss


# -----------------------
# Simple data augmentations
# -----------------------

def augment_audio(x: torch.Tensor, noise_std: float = 0.05, gain_range: tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    """
    Additive noise + random gain for audio.
    Args:
        x: [B, 1, T]
    """
    if x.dim() != 3:
        raise ValueError("augment_audio expects [B, 1, T]")
    noise = torch.randn_like(x) * noise_std
    gains = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(*gain_range)
    return (x + noise) * gains


def augment_imu(x: torch.Tensor, jitter_std: float = 0.02, axis_scale_std: float = 0.05) -> torch.Tensor:
    """
    Multiplicative per-axis scaling + additive jitter for IMU.
    Args:
        x: [B, 6, T] (accel xyz + gyro xyz)
    """
    if x.dim() != 3 or x.size(1) != 6:
        raise ValueError("augment_imu expects [B, 6, T]")
    # Per-sample, per-axis scale
    scales = (1.0 + axis_scale_std * torch.randn(x.size(0), 6, 1, device=x.device))
    jitter = torch.randn_like(x) * jitter_std
    return x * scales + jitter
