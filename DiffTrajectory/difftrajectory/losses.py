"""
Training objective, paper Eq. (16):

    L = omega * min_k||Y - Y_k||^2 + sum_k ||Y - Y_k||^2 / sigma_theta^2
        + lambda_1 ||mu - mu_prior||^2 + lambda_2 ||sigma^2 - sigma_prior^2||^2
        + log(sigma_theta^2)

  term 1 ("distance")      : best-of-K winner-take-all displacement loss
  term 2 ("decentralized")  : heteroscedastic-uncertainty term (a Gaussian
                              NLL in disguise: exp(-logvar)*error^2 + logvar)
  terms 3-4 ("weighted_smoothing") : LIM.prior_regularization, see models/lim.py
  term 5                    : log-variance regularizer preventing collapse

This is structurally identical to LED's training loss (see
trainer/train_led_trajectory_augment_input.py: `loss = loss_dist*50 +
loss_uncertainty`, with omega=50 matching Section 5.2's "we set the weight
parameter omega to 50" almost exactly) plus the two prior-regularization
terms DiffTrajectory's LIM adds on top.
"""
from dataclasses import dataclass

import torch

from .models.lim import LeapInitializerModule


@dataclass
class LossWeights:
    omega: float = 50.0     # Section 5.2: "we set the weight parameter omega to 50"
    lambda_1: float = 0.05  # Section 5.2
    lambda_2: float = 0.1   # Section 5.2
    use_temporal_reweight: bool = False  # LED applies this but calls it
                                          # "not necessary" in a code comment;
                                          # off by default here, flip on to
                                          # match LED's exact released config


def best_of_k_distance_loss(generated_y: torch.Tensor, fut_traj: torch.Tensor,
                             temporal_weight: torch.Tensor = None) -> torch.Tensor:
    """
    omega * min_k ||Y - Y_k||^2  term (the "omega *" scaling is applied by
    the caller, see `total_loss` below, matching how LED separates the two).

    generated_y : (B, K, T, d) -- K sampled future trajectories per scene
    fut_traj    : (B, T, d)    -- ground truth
    """
    err = (generated_y - fut_traj.unsqueeze(1)).norm(p=2, dim=-1)  # (B, K, T)
    if temporal_weight is not None:
        err = err * temporal_weight
    per_k = err.mean(dim=-1)          # (B, K)
    return per_k.min(dim=1)[0].mean()  # scalar


def uncertainty_loss(generated_y: torch.Tensor, fut_traj: torch.Tensor,
                      log_variance: torch.Tensor) -> torch.Tensor:
    """
    sum_k ||Y-Y_k||^2 / sigma_theta^2 + log(sigma_theta^2), collapsed to a
    mean the same way LED does (mean over K and T of the per-sample error,
    combined with a single scalar log-variance per scene).

    log_variance : (B, 1) -- the LIM's predicted log-scale for this scene
    """
    err = (generated_y - fut_traj.unsqueeze(1)).norm(p=2, dim=-1).mean(dim=(1, 2))  # (B,)
    inv_var = torch.exp(-log_variance.squeeze(-1))
    return (inv_var * err + log_variance.squeeze(-1)).mean()


def total_loss(generated_y: torch.Tensor, fut_traj: torch.Tensor,
                mean: torch.Tensor, log_variance: torch.Tensor,
                weights: LossWeights = LossWeights(),
                temporal_weight: torch.Tensor = None):
    """
    Full Eq. (16). Returns (total, dict-of-components) for logging.
    """
    dist = best_of_k_distance_loss(generated_y, fut_traj, temporal_weight)
    unc = uncertainty_loss(generated_y, fut_traj, log_variance)
    reg = LeapInitializerModule.prior_regularization(
        mean, log_variance, fut_traj, weights.lambda_1, weights.lambda_2
    )
    total = weights.omega * dist + unc + reg
    return total, {
        "distance": dist.item(),
        "uncertainty": unc.item(),
        "prior_reg": reg.item(),
        "total": total.item(),
    }
