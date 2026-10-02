"""
analyze_separation.py -- the cross-leakage regressions.  THE proof of separation.

  usage:  python analyze_separation.py            # all capture_icc_SEP_*.mat here

A low false-velocity RMSE on a static link proves nothing about whether Doppler
and oscillator offset are actually DECOUPLED -- with nothing injected, an
estimator that simply returns zero would score perfectly.  The real test is to
inject both quantities independently and check that each estimate responds to
its OWN driver and ignores the other:

    f_d_hat    = a_D  * f_D_GT     + b_D  * eps_LO_GT + c_D      want a_D  ~ 1, b_D  ~ 0
    eps_BV_hat = a_LO * eps_LO_GT  + b_LO * f_D_GT    + c_LO     want a_LO ~ 1, b_LO ~ 0

GROUND TRUTH AND SIGNS
  f_D_GT    = FD_INJECT_HW      -- pre-rotated on BOTH transmitters, so it does NOT
                                   flip with the TX/RX role: motion-like.
  eps_LO_GT = -LO_OFFSET_B_HZ   -- detuning node B's LO by +X lowers the DL CFO and
                                   raises the UL CFO, so eps_BV = (DL-UL)/2 moves by
                                   -X.  The minus sign is that convention, not a fudge.

c_LO absorbs the pair's NATURAL LO offset (~-2.9 kHz here), which is why only the
slopes matter.  One point per run: each run is one independent condition, so the
regression is not inflated by thousands of correlated frames within a run.
"""
import glob, os, sys
import numpy as np
from scipy.io import loadmat
from bidir_estimators import add_estimators

HERE  = os.path.dirname(os.path.abspath(__file__))
FILES = ([f for a in sys.argv[1:] for f in (sorted(glob.glob(a)) or [a])]   # expand globs (PowerShell does not)
        or sorted(glob.glob(os.path.join(HERE, "..", "data", "captures", "capture_icc_SEP_*.mat"))))
SKIP  = 5          # drop this many opening exchanges per run (settling)
# Which Doppler estimate to test.  fd_driftcomp is the default because it needs no
# calibration -- important here, since every injected condition would otherwise
# carry a bias constant measured under a different drift state.  Alternatives:
# 'fd_plain' | 'fd_tinterp' | 'fd_biascal'  (legacy: 'fd_half' | 'fd_ta' | 'fd_cal')
EST   = 'fd_driftcomp'

if not FILES:
    sys.exit("No capture_icc_SEP_*.mat found.  Run:  python run_matrix.py sep")

rows = []
for fn in FILES:
    d = add_estimators(loadmat(fn, squeeze_me=True))
    missing = [k for k in (EST, 'eps_bv', 'FD_INJECT_HW', 'HZ_PER_MPS') if k not in d]
    if missing:
        print(f"  skipping {os.path.basename(fn)} (missing {', '.join(missing)})"); continue
    fd_gt  = float(d['FD_INJECT_HW'])
    lo_gt  = -float(d.get('LO_OFFSET_B_HZ', 0.0))      # see sign note above
    fd_est = np.atleast_1d(d[EST]).astype(float)
    eps    = np.atleast_1d(d['eps_bv']).astype(float)
    m      = np.isfinite(fd_est)
    m[:SKIP] = False
    if m.sum() < 5:
        print(f"  skipping {os.path.basename(fn)} (only {m.sum()} usable)"); continue
    rows.append((fd_gt, lo_gt, float(np.mean(fd_est[m])), float(np.std(fd_est[m])),
                 float(np.mean(eps[m])), float(np.std(eps[m])),
                 int(m.sum()), str(d.get('RUN_TAG', os.path.basename(fn)))))

if len(rows) < 4:
    sys.exit(f"Only {len(rows)} conditions -- need the full matrix for a regression.")

# --- did the randomisation work?  The run order is in the tag as _sNN_.  If the
# estimate still correlates with WHEN a condition ran, drift is leaking in and the
# slopes below are not trustworthy.  This is the check the old fixed-order sweep
# could never have passed.
import re as _re
_seq = [(_re.search(r'_s(\d+)_', r[7]) or [None, None])[1] for r in rows]
if all(x is not None for x in _seq):
    _sq = np.array([float(x) for x in _seq])
    _fh = np.array([r[2] - r[0] for r in rows])      # residual, not the injected value itself
    if _sq.size > 3 and _sq.std() > 0:
        _c = float(np.corrcoef(_sq, _fh)[0, 1])
        _v = "OK" if abs(_c) < 0.3 else "<-- DRIFT IS STILL LEAKING"
        print("")
        print(f"  run-order check: corr(sequence index, f_d hat - f_D GT) = {_c:+.3f}   {_v}")

fd_gt, lo_gt, fd_hat, fd_sd, eps_hat, eps_sd, ns, tags = map(np.array, zip(*rows))

print("\n" + "="*94)
print(f"SEPARATION MATRIX -- {len(rows)} conditions, estimate '{EST}'")
print("="*94)
print(f"{'run':<22}{'f_D GT':>10}{'eps_LO GT':>11}{'f_D hat':>10}{'sd':>8}"
      f"{'eps_BV hat':>12}{'sd':>8}{'n':>5}")
for i in np.argsort(lo_gt*1e6 + fd_gt):
    print(f"{tags[i]:<22}{fd_gt[i]:>10.2f}{lo_gt[i]:>11.0f}{fd_hat[i]:>10.3f}"
          f"{fd_sd[i]:>8.3f}{eps_hat[i]:>12.1f}{eps_sd[i]:>8.2f}{ns[i]:>5d}")

def fit(y, x1, x2, names):
    """Least squares y = a*x1 + b*x2 + c, with standard errors."""
    A = np.column_stack([x1, x2, np.ones(x1.size)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof   = max(A.shape[0] - A.shape[1], 1)
    s2    = float(resid @ resid)/dof
    cov   = s2*np.linalg.pinv(A.T @ A)
    se    = np.sqrt(np.diag(cov))
    ss_tot = float(((y - y.mean())**2).sum())
    r2 = 1.0 - float(resid @ resid)/ss_tot if ss_tot > 0 else np.nan
    for nm, b, e in zip(names, beta, se):
        print(f"    {nm:<28}{b:+12.5f}  +/- {e:.5f}")
    print(f"    {'R^2':<28}{r2:+12.5f}")
    return beta, se

print("\n--- Doppler estimate vs both drivers " + "-"*54)
print("    (want: slope on f_D ~ +1, slope on eps_LO ~ 0)")
bD, seD = fit(fd_hat, fd_gt, lo_gt,
              ["a_D  (slope on f_D_GT)", "b_D  (LEAKAGE from eps_LO)", "c_D  (offset, Hz)"])

print("\n--- LO estimate vs both drivers " + "-"*58)
print("    (want: slope on eps_LO ~ +1, slope on f_D ~ 0)")
bL, seL = fit(eps_hat, lo_gt, fd_gt,
              ["a_LO (slope on eps_LO_GT)", "b_LO (LEAKAGE from f_D)", "c_LO (natural offset, Hz)"])

print("\n--- verdict " + "-"*78)
# A dimensionless leakage coefficient is not interpretable on its own, so convert it
# into the error it actually causes on THIS hardware: the pair's own natural LO offset
# times the leakage, in m/s, judged against the measured run-to-run noise floor.
HZ_MPS  = float(loadmat(FILES[0], squeeze_me=True)['HZ_PER_MPS'])
eps_nat = abs(bL[2])                        # c_LO = this pair's natural LO offset (Hz)
fd_span = float(np.max(np.abs(fd_gt))) or 1.0
floor_v = float(np.median(fd_sd))/HZ_MPS    # within-run velocity scatter, m/s
leak_v  = abs(bD[1])*eps_nat/HZ_MPS         # false velocity from LO leakage, m/s
leak_lo = abs(bL[1])*fd_span                # false LO from Doppler leakage, Hz

print(f"    natural LO offset of this pair : {eps_nat:8.1f} Hz")
print(f"    within-run velocity noise floor: {floor_v:8.4f} m/s")
print(f"    a_D  Doppler gain              : {bD[0]:+8.4f} +/- {seD[0]:.4f}  (want +1)")
print(f"    a_LO LO gain                   : {bL[0]:+8.4f} +/- {seL[0]:.4f}  (want +1)")
print(f"    b_D  LO->Doppler leakage       : {bD[1]:+8.5f}  =>  {leak_v:.4f} m/s of")
print(f"         FALSE VELOCITY caused by the natural offset alone")
print(f"    b_LO Doppler->LO leakage       : {bL[1]:+8.5f}  =>  {leak_lo:.3f} Hz of")
print(f"         LO error at the largest injected Doppler")

ok_gain = abs(bD[0]-1) <= max(0.05, 2*seD[0]) and abs(bL[0]-1) <= max(0.05, 2*seL[0])
ok_leak = leak_v <= floor_v                 # leakage must sit UNDER the noise floor
print(f"\n    gains within 5% of unity        : {'PASS' if ok_gain else 'FAIL'}")
print(f"    LO leakage below the noise floor: "
      f"{'PASS' if ok_leak else 'FAIL'}  ({leak_v:.4f} vs {floor_v:.4f} m/s)")
print("\n    " + ("SEPARATION DEMONSTRATED" if (ok_gain and ok_leak) else
      "NOT clean yet -- the two lines above say which half fails"))

# ---- the two headline plots ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\n  (matplotlib not installed -- run: pip install matplotlib)")
else:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for lo in np.unique(lo_gt):
        k = lo_gt == lo
        o = np.argsort(fd_gt[k])
        ax[0].errorbar(fd_gt[k][o], fd_hat[k][o], yerr=fd_sd[k][o], marker='o',
                       capsize=3, lw=1.2, label=f"eps_LO = {lo:+.0f} Hz")
        ax[1].errorbar(fd_gt[k][o], eps_hat[k][o], yerr=eps_sd[k][o], marker='s',
                       capsize=3, lw=1.2, label=f"eps_LO = {lo:+.0f} Hz")
    lim = [fd_gt.min()-10, fd_gt.max()+10]
    ax[0].plot(lim, lim, 'k--', lw=1, label='ideal (slope 1)')
    ax[0].set_xlabel('injected Doppler $f_D^{GT}$ [Hz]')
    ax[0].set_ylabel(r'estimated $\hat{f}_D$ [Hz]')
    ax[0].set_title('Doppler tracks Doppler, not the LO offset')
    ax[1].set_xlabel('injected Doppler $f_D^{GT}$ [Hz]')
    ax[1].set_ylabel(r'estimated $\hat{\epsilon}_{BV}$ [Hz]')
    ax[1].set_title('LO estimate is flat vs Doppler')
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    out = os.path.join(HERE, "fig_separation.png")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\n  figure -> {out}")
print()
