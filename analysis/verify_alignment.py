"""
verify_alignment.py -- does epoch alignment actually cancel the oscillator term?

The claim in Section III-B is structural: if the DL observation is moved to the
UL epoch, the term [eps(t_U) - eps(t_D)]/2 cancels EXACTLY, for any eps(t), not
just to first order.  Drift compensation instead cancels it only to the extent
that a slope fitted over ~21 exchanges predicts the slope during the gap.

This is a controlled test of that claim.  A STATIC capture (true Doppler zero)
is taken, and a SYNTHETIC oscillator term eps_syn(t) is added to the DL and
subtracted from the UL -- exactly how a real oscillator enters (4)-(5).  The
real data keeps its own noise, so differencing against the un-augmented run
isolates what each estimator does with eps_syn alone.

If the derivation is right:
    raw half sum     -> picks up  -[eps(t_U)-eps(t_D)]/2, i.e. about -r*Delta/2
    drift compensated-> removes it only while eps is locally linear over the window
    epoch aligned    -> removes it whatever eps does, up to the curvature of the
                        INTERPOLATION over one DL update interval

    python verify_alignment.py ../data/captures/capture_icc_WARM_0922_0807.mat
"""
import sys
import numpy as np
from icc_common import load
from bidir_estimators import get_variant, local_drift, DRIFT_WINDOW, MAX_SPAN_MULT

CAP = sys.argv[1] if len(sys.argv) > 1 else '../data/captures/capture_icc_WARM_0922_0807.mat'


def estimators(nu_d, nu_u, t_d, t_u, gap):
    """The three Section III estimators, from the raw per-direction CFOs."""
    half = (nu_d + nu_u) / 2.0
    f_lo = (nu_d - nu_u) / 2.0
    D = local_drift(t_d, f_lo, DRIFT_WINDOW)
    dcomp = half + D * gap / 2.0
    # epoch alignment: interpolate the DL to the UL epoch
    span = np.diff(t_d, append=np.nan)
    a = (t_u - t_d) / span
    nu_d_at_u = np.full(nu_d.size, np.nan)
    ok = np.isfinite(span) & (span <= MAX_SPAN_MULT * np.nanmedian(span)) & (a > -0.5) & (a < 1.5)
    ok[-1] = False
    nu_d_at_u[ok] = (1 - a[ok]) * nu_d[:-1][ok[:-1]] if False else np.nan  # placeholder
    idx = np.where(ok)[0]
    nu_d_at_u[idx] = (1 - a[idx]) * nu_d[idx] + a[idx] * nu_d[idx + 1]
    align = (nu_d_at_u + nu_u) / 2.0
    return half, dcomp, align


d = load(CAP)
t_d = np.asarray(get_variant(d, 't_A2B_rel'), float)
t_u = t_d + np.asarray(get_variant(d, 'gap_s'), float)
gap = t_u - t_d
nu_d0 = np.asarray(get_variant(d, 'cfo_A2B'), float)
nu_u0 = np.asarray(get_variant(d, 'cfo_B2A'), float)
sB = np.asarray(get_variant(d, 'snr_at_B'), float)
sA = np.asarray(get_variant(d, 'snr_at_A'), float)
ok = np.isfinite(nu_d0) & np.isfinite(nu_u0) & (sB > 8) & (sA > 8)
hz = float(d['HZ_PER_MPS'])
T = t_d - t_d[0]

base = estimators(nu_d0, nu_u0, t_d, t_u, gap)

print(f"{CAP}   {ok.sum()} usable exchanges, median gap {1e3*np.nanmedian(gap):.0f} ms, "
      f"median DL update {1e3*np.nanmedian(np.diff(t_d)):.0f} ms")
print("\nsynthetic oscillator term eps_syn(t) added to DL, subtracted from UL.")
print("reported: RMS of the CHANGE each estimator shows, in m/s (true change = 0)\n")
print(f"  {'eps_syn(t)':<34}{'raw half sum':>14}{'drift comp.':>14}{'epoch aligned':>15}")

rng = np.random.default_rng(0)
# eps_syn is a CONTINUOUS function of time, sampled independently at t_d and t_u.
# Building the UL sample by interpolating the DL grid would assume exactly what the
# aligner assumes and make the test circular.
t0 = t_d[0]
fine = np.arange(0.0, max(t_d[-1], t_u[-1]) - t0 + 1.0, 1e-3)   # 1 ms grid
rw = np.cumsum(rng.normal(0.0, 2.0 * np.sqrt(1e-3), fine.size))
tmid = t_d[len(t_d) // 2] - t0
cases = [
    ("linear ramp, 1 Hz/s",             lambda x: 1.0 * x),
    ("linear ramp, 10 Hz/s",            lambda x: 10.0 * x),
    ("quadratic, 1 Hz/s^2",             lambda x: 0.5 * x ** 2),
    ("sine, 1 Hz ampl, 60 s period",    lambda x: np.sin(2 * np.pi * x / 60)),
    ("sine, 1 Hz ampl, 10 s period",    lambda x: np.sin(2 * np.pi * x / 10)),
    ("sine, 1 Hz ampl, 2 s period",     lambda x: np.sin(2 * np.pi * x / 2)),
    ("sine, 1 Hz ampl, 0.5 s period",   lambda x: np.sin(2 * np.pi * x / 0.5)),
    ("random walk, 2 Hz/sqrt(s)",       lambda x: np.interp(x, fine, rw)),
    ("step of 10 Hz at mid-run",        lambda x: 10.0 * (x > tmid)),
]
for name, fn in cases:
    aug = estimators(nu_d0 + fn(t_d - t0), nu_u0 - fn(t_u - t0), t_d, t_u, gap)
    row = []
    for b, a_ in zip(base, aug):
        delta = (a_ - b) / hz
        m = ok & np.isfinite(delta)
        row.append(np.sqrt(np.mean(delta[m] ** 2)))
    print(f"  {name:<34}{row[0]:>14.4f}{row[1]:>14.4f}{row[2]:>15.4f}")

print("\n(the last column is what the derivation predicts should stay at zero)")
