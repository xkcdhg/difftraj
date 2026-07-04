"""
Generic fixed-step ODE solvers: Euler, Heun (RK2), and classical RK4.

These implement the numerical methods compared in Section 3.2 / 4.1 of
DiffTrajectory (Li et al., Pattern Recognition 172 (2026) 112339):

    Method   Order   LTE      GTE          (paper's Table 1)
    Euler    1st     O(h^2)   O(h)
    Heun     2nd     O(h^3)   O(h^2)
    RK4      4th     O(h^5)   O(h^4)

Design note
-----------
This module solves the *general* initial value problem dx/dt = F(x, t) and
steps x(t) -> x(t + dt) for an arbitrary (possibly negative) dt. It has no
notion of "denoising" or "diffusion time" baked in -- that's the job of
`difftrajectory/diffusion.py`, which wraps a trained noise-prediction network
eps_theta into an F(x, t) callable and calls these steppers with dt = -delta,
since the reverse/denoising process runs t from Gamma down to 0.

x can be a NumPy array or a PyTorch tensor: both support +, -, *, / against
python scalars and against each other, so this file has *zero* hard
dependency on either library. That's what lets us unit-test RK4's 4th-order
convergence on toy problems using nothing but NumPy (see tests/), which we
can run right here in a CPU-only sandbox without installing PyTorch or
touching any real trajectory data.

A note on the paper's Eq. (10)-(11) sign convention
----------------------------------------------------
The paper writes the RK4 stages with *plus* signs (x_tn + k1/2, t_n + delta/2,
...) and then *subtracts* the weighted combination to obtain x_{tn-delta}.
Taken completely literally, that mixes a forward-stepping stage evaluation
with a backward final update, which is not the standard (self-consistent)
RK4 recurrence for integrating dx/dt = F(x,t) backward in t. We believe this
is a typesetting/notational simplification rather than the authors' actual
implementation (the paper has a handful of other typos, e.g. Algorithm 1
uses `lambda` where the surrounding prose clearly means `xi`, and Section 2.1
has a duplicated word) -- and it is the kind of sign slip that would silently
still "run" while quietly hurting accuracy, so we don't want to copy it
without comment.

`rk4_step` below therefore implements the standard, mathematically
self-consistent recurrence for a *signed* dt (positive = forward, negative =
backward), which reduces to the paper's Eq. (11) with a sign flip on the
intermediate stages when dt = -delta. This is a judgment call -- flagged
here and in docs/PLAN.md -- that should be revisited once real training
lets us compare against the paper's reported numbers.
"""
from typing import Callable, Dict, TypeVar

State = TypeVar("State")                    # np.ndarray or torch.Tensor
DriftFn = Callable[[State, float], State]   # F(x, t) -> dx/dt


def euler_step(F: DriftFn, x: State, t: float, dt: float) -> State:
    """First-order Euler step (paper Eq. 4). LTE = O(dt^2), GTE = O(dt)."""
    return x + dt * F(x, t)


def heun_step(F: DriftFn, x: State, t: float, dt: float) -> State:
    """
    Second-order Heun / improved-Euler predictor-corrector step
    (paper Eq. 6-7). LTE = O(dt^3), GTE = O(dt^2).
    """
    k1 = F(x, t)
    x_pred = x + dt * k1                 # Euler predictor
    k2 = F(x_pred, t + dt)
    return x + (dt / 2.0) * (k1 + k2)    # trapezoidal correction


def rk4_step(F: DriftFn, x: State, t: float, dt: float) -> State:
    """
    Classical fourth-order Runge-Kutta step (paper Eq. 9-11).
    LTE = O(dt^5), GTE = O(dt^4) (Table 1).

    Parameters
    ----------
    F  : callable(x, t) -> dx/dt
    x  : current state
    t  : current "time" (diffusion noise level)
    dt : signed step. Pass dt = -delta to move from noise level t to
         t - delta (a denoising step, paper's x_{tn-delta}); pass
         dt = +delta to integrate forward.
    """
    k1 = dt * F(x, t)
    k2 = dt * F(x + k1 / 2.0, t + dt / 2.0)
    k3 = dt * F(x + k2 / 2.0, t + dt / 2.0)
    k4 = dt * F(x + k3, t + dt)
    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


STEPPERS: Dict[str, Callable[..., State]] = {
    "euler": euler_step,
    "heun": heun_step,
    "rk4": rk4_step,
}
