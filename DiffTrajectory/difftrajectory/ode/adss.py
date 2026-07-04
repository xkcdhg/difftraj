"""
ADSS: Adaptive Dynamic Step-size Strategy (paper Section 4.3, Algorithm 1).

Reads the paper's prose (which is more internally consistent than the
typeset Algorithm 1 box -- see note below) as a classic embedded
predictor-corrector step-size controller:

  1. At the current state x_tau, try a candidate step size delta_p.
  2. Get a *cheap* local-error estimate using a first-order Euler predictor
     and a second-order Heun corrector (2 network evals, reusing one):
         x_pred = Euler-step(x_tau, tau, delta_p)
         x_corr = Heun-step (x_tau, tau, delta_p)
         zeta   = || x_pred - x_corr ||
     (this is the same idea as classical embedded RK error control, e.g.
     Runge-Kutta-Fehlberg / Dormand-Prince, just using a 1st/2nd order pair
     instead of two different high-order formulas.)
  3. If zeta < xi (tolerance): ACCEPT. Take the *actual* denoising step with
     RK4 (4 evals, high accuracy) at size delta_p, advance tau -= delta_p,
     and GROW the candidate step size for next round:
         delta_next = delta_p * (xi / zeta) ** (1 / gamma)
  4. Else: REJECT. Shrink the candidate step size with the same formula
     (now xi/zeta < 1, so it shrinks) and retry from step 2 without
     advancing tau.

This is why the ablation in Table 6 reports needing only ~2-3 *steps* to
reach tau=0 with ADSS instead of a fixed 10 RK4 steps: most iterations
succeed at a large step, and the per-iteration error check is cheap
(2 evals) relative to committing an RK4 step (4 evals).

Note on the typeset Algorithm 1 box vs. this implementation
-------------------------------------------------------------
The paper's Algorithm 1 pseudocode has a couple of internal inconsistencies
against its own prose (e.g. step 8 uses the symbol `lambda`, which is never
defined anywhere else in the paper and is almost certainly a typo for `xi`;
the rectification formula in step 5 has a garbled argument
`F^p(x_tau + delta, tau + delta_p)` that doesn't parse against the prose,
which clearly describes a standard Heun corrector evaluated at the Euler
prediction). We implement the prose description, which is internally
self-consistent and matches a well-known numerical recipe. This is a
judgment call, flagged here and in docs/PLAN.md, to revisit once we can
compare against the paper's actual reported numbers.
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple, TypeVar

from .solvers import rk4_step, euler_step, heun_step

State = TypeVar("State")
DriftFn = Callable[[State, float], State]


def _norm(x) -> float:
    """L2 norm that works for both a python/numpy array and a torch tensor
    without importing either at module load time. Always detached: zeta is
    used purely as a scalar control signal for the accept/reject decision
    and step-size formula, never as something to backprop through."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return float(x.detach().norm(p=2))
    except ImportError:
        pass
    import numpy as np
    return float(np.linalg.norm(x))


@dataclass
class ADSSConfig:
    delta_init: float = 3.0     # paper Sec 5.2: "initial adaptive step size ... value of 3"
    delta_min: float = 0.5
    delta_max: float = 10.0
    xi: float = 0.05            # acceptable error tolerance
    gamma: float = 0.6          # step-scaling exponent base (paper sweeps 0.1-1.0, Table 8;
                                 # best inference-time/accuracy tradeoff around 0.5-0.6)
    max_reject_retries: int = 25
    max_total_steps: int = 200  # hard safety cap: never let a stiff region hang the loop
    eps: float = 1e-8           # guards against division by zero when zeta == 0


def adss_denoise(
    F: DriftFn,
    x: State,
    tau: float,
    cfg: ADSSConfig,
) -> Tuple[State, List[Tuple[float, float, float]]]:
    """
    Run the full ADSS loop from diffusion time `tau` down to 0, starting
    from state `x` (e.g. the output of the LIM leap-initializer).

    Returns
    -------
    x       : final denoised state at tau=0
    trace   : list of (tau_after_step, step_size_taken, zeta) for
              inspection/debugging/plotting -- e.g. to reproduce a Fig. 1(b)
              -style visualization of skipped step sizes.
    """
    delta_p = min(cfg.delta_init, tau)
    trace: List[Tuple[float, float, float]] = []
    total_steps = 0

    while tau > 0 and total_steps < cfg.max_total_steps:
        delta_p = min(max(delta_p, cfg.delta_min), tau)
        zeta = None
        for _ in range(cfg.max_reject_retries):
            x_pred = euler_step(F, x, tau, -delta_p)
            x_corr = heun_step(F, x, tau, -delta_p)
            zeta = _norm(_sub(x_pred, x_corr))

            if zeta < cfg.xi:
                # Accept: grow the step for the *next* round.
                delta_next = delta_p * (cfg.xi / max(zeta, cfg.eps)) ** (1.0 / cfg.gamma)
                delta_next = min(max(delta_next, cfg.delta_min), cfg.delta_max)
                break
            elif delta_p <= cfg.delta_min + 1e-12:
                # Already at the floor and still over tolerance: take it
                # anyway (can't do better) instead of spinning forever.
                delta_next = cfg.delta_min
                break
            else:
                # Reject: shrink the candidate (never below delta_min) and
                # retry the error check.
                delta_p = delta_p * (cfg.xi / max(zeta, cfg.eps)) ** (1.0 / cfg.gamma)
                delta_p = max(delta_p, cfg.delta_min)
                delta_p = min(delta_p, tau)
        else:
            # Safety valve: exhausted retries, just go with the floor.
            delta_next = cfg.delta_min

        step = min(delta_p, tau)
        x = rk4_step(F, x, tau, -step)
        tau = tau - step
        total_steps += 1
        trace.append((tau, step, zeta))
        delta_p = delta_next

    return x, trace


def _sub(a, b):
    return a - b
