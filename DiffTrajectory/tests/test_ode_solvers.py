"""
Validate that euler_step / heun_step / rk4_step achieve their theoretical
global convergence orders (paper's Table 1: O(h), O(h^2), O(h^4)) on a toy
ODE with a known closed-form solution.

This deliberately uses plain NumPy floats, not the trajectory-prediction
score network, so it can run anywhere with zero ML dependencies -- it is a
pure correctness check of the numerical integrators in isolation.

Toy problem: dx/dt = -x,  x(0) = 1  =>  exact solution x(t) = exp(-t).
We integrate forward from t=0 to t=1 with N fixed steps of size h = 1/N,
for several step counts, and check that halving h reduces the global error
by ~2x (Euler), ~4x (Heun), ~16x (RK4).
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from difftrajectory.ode.solvers import euler_step, heun_step, rk4_step  # noqa: E402


def integrate(stepper, F, x0, t0, t1, n_steps):
    h = (t1 - t0) / n_steps
    x, t = x0, t0
    for _ in range(n_steps):
        x = stepper(F, x, t, h)
        t += h
    return x


def F(x, t):
    return -x  # dx/dt = -x


def measure_orders(stepper, step_counts=(10, 20, 40, 80, 160)):
    exact = math.exp(-1.0)
    errors = []
    for n in step_counts:
        x_final = integrate(stepper, F, 1.0, 0.0, 1.0, n)
        errors.append(abs(x_final - exact))
    # empirical order between consecutive (h, h/2) pairs:
    # order ~= log2(error(h) / error(h/2))
    orders = [
        math.log2(errors[i] / errors[i + 1]) if errors[i + 1] > 0 else float("inf")
        for i in range(len(errors) - 1)
    ]
    return errors, orders


def test_convergence_orders():
    results = {}
    for name, stepper, expected_order in [
        ("euler", euler_step, 1),
        ("heun", heun_step, 2),
        ("rk4", rk4_step, 4),
    ]:
        errors, orders = measure_orders(stepper)
        results[name] = (errors, orders)
        print(f"\n{name:5s}  expected order ~{expected_order}")
        for n, e in zip((10, 20, 40, 80, 160), errors):
            print(f"  n={n:4d} steps  |error| = {e:.3e}")
        print(f"  empirical orders between successive halvings: "
              f"{[f'{o:.2f}' for o in orders]}")
        # The last (finest) empirical order estimate should be close to the
        # theoretical order -- allow generous tolerance since this is a
        # smoke test, not a numerical-analysis paper.
        assert abs(orders[-1] - expected_order) < 0.3, (
            f"{name}: expected order ~{expected_order}, "
            f"got empirical order {orders[-1]:.2f}"
        )
    return results


def test_rk4_much_more_accurate_than_euler_at_equal_cost():
    """Sanity check mirroring the paper's motivation: at a *fixed* number of
    steps (i.e. fixed compute budget), RK4 should be dramatically more
    accurate than Euler, even though each RK4 step costs 4x the function
    evaluations of one Euler step."""
    exact = math.exp(-1.0)
    n = 10
    euler_err = abs(integrate(euler_step, F, 1.0, 0.0, 1.0, n) - exact)
    rk4_err = abs(integrate(rk4_step, F, 1.0, 0.0, 1.0, n) - exact)
    print(f"\nAt n={n} steps: Euler error={euler_err:.3e}, RK4 error={rk4_err:.3e}, "
          f"ratio={euler_err / rk4_err:.1f}x")
    assert rk4_err < euler_err / 100  # RK4 should be >100x more accurate here


def test_backward_integration_is_consistent_with_forward():
    """Integrating forward then backward by the same steps should return
    (approximately) to the start -- validates the dt-sign convention used
    for the denoising direction (dt < 0) described in solvers.py."""
    x0 = 1.0
    x_fwd = integrate(rk4_step, F, x0, 0.0, 1.0, 50)

    # now integrate backward from t=1 to t=0
    h = -1.0 / 50
    x, t = x_fwd, 1.0
    for _ in range(50):
        x = rk4_step(F, x, t, h)
        t += h
    print(f"\nRound-trip via RK4 (forward then backward): "
          f"start={x0}, after round trip={x:.8f}")
    assert abs(x - x0) < 1e-6


if __name__ == "__main__":
    test_convergence_orders()
    test_rk4_much_more_accurate_than_euler_at_equal_cost()
    test_backward_integration_is_consistent_with_forward()
    print("\nAll ODE solver tests passed.")
