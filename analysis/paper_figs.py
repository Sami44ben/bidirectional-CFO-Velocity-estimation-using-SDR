"""paper_figs.py -- Python previews of the separation and gap figures, as vector PDF.

    python paper_figs.py            # writes ../paper/figures/fig_{separation,gap}_py.pdf

THE PAPER FIGURES ARE DRAWN BY matlab/fig_*_m.m (with editable .fig files).  This
script writes *_py.pdf so it can never overwrite them.  No tare (c = 0).

Reads the captures named below; every number in the figures is recomputed
from raw per-direction CFOs through bidir_estimators, nothing is typed in.
"""
import glob, os, re
import numpy as np
from scipy.io import loadmat
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bidir_estimators import add_estimators

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "paper", "figures")
# Session stamps are read from the environment so a re-run never needs an edit:
#   ICC_SEP=0921_1500 ICC_GAP_WARM=0921_1530 ICC_GAP_COLD=0922_0900 ICC_WARM=0921_1440 python paper_figs.py
# Unset, each falls back to the NEWEST stamp present in data/captures/.
def _latest(prefix):
    fs = sorted(glob.glob(os.path.join(HERE, "..", "data", "captures", f"capture_icc_{prefix}_*.mat")), key=os.path.getmtime)
    if not fs:
        raise SystemExit(f"no ../data/captures/capture_icc_{prefix}_*.mat")
    return re.search(rf"{prefix}_(\d{{4}}_\d{{4}})", os.path.basename(fs[-1])).group(1)
_sep  = os.environ.get("ICC_SEP")      or _latest("SEP")
_gw   = os.environ.get("ICC_GAP_WARM") or _latest("GAP")
_gc   = os.environ.get("ICC_GAP_COLD") or _gw
_warm = os.environ.get("ICC_WARM")     or _latest("WARM")
SEP  = f"../data/captures/capture_icc_SEP_{_sep}_*.mat"
GAPS = {"warm radios": f"../data/captures/capture_icc_GAP_{_gw}_*.mat"}
if _gc != _gw:
    GAPS["cold start"] = f"../data/captures/capture_icc_GAP_{_gc}_*.mat"
TARE = f"../data/captures/capture_icc_WARM_{_warm}.mat"          # static run the tare comes from
print(f"sessions: SEP {_sep} | GAP warm {_gw} | GAP cold {_gc} | WARM {_warm}")
SKIP = 5                                            # settling exchanges dropped per run
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": .3})

load = lambda f: add_estimators(loadmat(f, squeeze_me=True))
w = load(TARE); c = 0.0; hz = float(w['HZ_PER_MPS'])        # no tare (beta_+ = 0.01 Hz, below the floor)

# ---- Fig: separation matrix ----------------------------------------------
rows = []
for f in sorted(glob.glob(os.path.join(HERE, SEP))):
    d = load(f); k = np.arange(d['exch_id'].size) >= SKIP
    rows.append((float(d['FD_INJECT_HW']), -float(d['LO_OFFSET_B_HZ']),
                 np.mean(d['fd_driftcomp'][k]), np.std(d['fd_driftcomp'][k]),
                 np.mean(d['f_lo_offset'][k]), np.std(d['f_lo_offset'][k])))
fd, lo, fh, fs, eh, es = map(np.array, zip(*rows))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for l, mk in zip(np.unique(lo), "osv"):
    k = lo == l; o = np.argsort(fd[k])
    ax[0].errorbar(fd[k][o]/hz, fh[k][o]/hz, yerr=fs[k][o]/hz, marker=mk, ms=4, capsize=2, lw=1, label=f"$\\epsilon^{{\\rm inj}}={l:+.0f}$ Hz")
    ax[1].errorbar(fd[k][o]/hz, eh[k][o], yerr=es[k][o], marker=mk, ms=4, capsize=2, lw=1, label=f"$\\epsilon^{{\\rm inj}}={l:+.0f}$ Hz")
ax[0].plot([-11, 11], [-11, 11], 'k--', lw=.8, label="unit slope")
ax[0].set(xlabel="injected velocity [m/s]", ylabel="estimated velocity [m/s]")
ax[1].set(xlabel="injected velocity [m/s]", ylabel=r"estimated $\hat\epsilon$ [Hz]")
ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_separation_py.pdf")); plt.close(fig)

# ---- Fig: gap sweep --------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
for (lab, pat), col in zip(GAPS.items(), ("C0", "C3")):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, pat))):
        d = load(f); k = np.arange(d['exch_id'].size) >= SKIP
        g = np.median(d['gap_s'][k]); D = np.polyfit(d['t_dl_rel'][k], d['f_lo_offset'][k], 1)[0]
        r = lambda v: np.sqrt(np.nanmean(v[k]**2))/hz
        rows.append((g, -D*g/2, np.mean(d['fd_plain'][k]) - c, r(d['fd_plain']-c), r(d['fd_driftcomp']-c), r(d['fd_tinterp']-c)))
    if not rows: continue
    g, pred, meas, rp, rd, rt = map(np.array, zip(*sorted(rows)))
    ax[0].plot(pred, meas, 'o', color=col, ms=4, label=lab)
    for gi, pi, mi in zip(g, pred, meas): ax[0].annotate(f"{gi:.1f}", (pi, mi), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax[1].plot(g, rp, 'o-', color=col, ms=3, lw=1, label=f"{lab}: half-sum")
    ax[1].plot(g, rd, 's--', color=col, ms=3, lw=1, label=f"{lab}: drift-comp.")
    ax[1].plot(g, rt, '^:', color=col, ms=3, lw=1, label=f"{lab}: interpolated")
lim = np.array(ax[0].get_xlim()); ax[0].plot(lim, lim, 'k--', lw=.8, label="measured = predicted")
ax[0].set(xlabel=r"predicted bias $-\dot\epsilon\Delta/2$ [Hz]", ylabel="measured half-sum shift [Hz]")
ax[1].set(xlabel=r"DL--UL gap $\Delta$ [s]", ylabel="RMSE [m/s]")
ax[0].legend(fontsize=7); ax[1].legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_gap_py.pdf")); plt.close(fig)
print("wrote", os.path.abspath(OUT))
