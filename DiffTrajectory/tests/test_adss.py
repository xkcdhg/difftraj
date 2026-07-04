"""
Sanity checks for ADSS (difftrajectory/ode/adss.py) on toy problems, mirroring
the qualitative claims in the paper (Table 6/8/9): ADSS should reach tau=0
in far fewer *steps* than a fixed step size, by taking large steps where the
local error is small.
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from difftrajectory.ode.adss import ADSSConfig, adss_denoise  # noqa: E402


def test_adss_terminates_and_hits_zero():
    F = lambda x, t: -0.1 * x  # gentle linear drift -> low error at large steps
    cfg = ADSSConfig(delta_init=3.0, delta_min=0.1, delta_max=10.0, xi=0.05, gamma=0.6)
    x_final, trace = adss_denoise(F, x=1.0, tau=10.0, cfg=cfg)
    taus = [t for t, _, _ in trace]
    print(f"\nGentle drift: {len(trace)} steps taken, final tau={taus[-1]:.6f}, "
          f"step sizes={[round(s, 2) for _, s, _ in trace]}")
    assert abs(taus[-1]) < 1e-9, "ADSS should land exactly on tau=0"
    assert len(trace) <= 10, "should need far fewer than 10 unit steps for an easy ODE"


def test_adss_takes_more_steps_on_stiff_problem():
    """A drift with large curvature should force smaller accepted steps than
    a gentle one, since the Euler/Heun local-error estimate will be larger."""
    gentle = lambda x, t: -0.1 * x
    stiff = lambda x, t: -5.0 * x - 2.0 * math.tanh(x) ** 3

    cfg = ADSSConfig(delta_init=3.0, delta_min=0.05, delta_max=10.0, xi=0.02, gamma=0.6)
    _, trace_gentle = adss_denoise(gentle, x=1.0, tau=10.0, cfg=cfg)
    _, trace_stiff = adss_denoise(stiff, x=1.0, tau=10.0, cfg=cfg)

    print(f"\nGentle problem: {len(trace_gentle)} steps; "
          f"Stiff problem: {len(trace_stiff)} steps")
    assert len(trace_stiff) >= len(trace_gentle), (
        "ADSS should need more (smaller) steps on the higher-curvature problem"
    )


def test_adss_respects_step_bounds():
    F = lambda x, t: -0.01 * x  # extremely gentle -> would want huge steps
    cfg = ADSSConfig(delta_init=1.0, delta_min=0.1, delta_max=2.0, xi=0.1, gamma=0.5)
    _, trace = adss_denoise(F, x=1.0, tau=10.0, cfg=cfg)
    for _, step, _ in trace:
        assert step <= cfg.delta_max + 1e-9, f"step {step} exceeded delta_max"
    print(f"\nAll {len(trace)} accepted steps respected delta_max={cfg.delta_max}: "
          f"max step taken={max(s for _, s, _ in trace):.3f}")


if __name__ == "__main__":
    test_adss_terminates_and_hits_zero()
    test_adss_takes_more_steps_on_stiff_problem()
    test_adss_respects_step_bounds()
    print("\nAll ADSS tests passed.")
