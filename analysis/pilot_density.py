"""
pilot_density.py -- precision of the half sum against its closed form, measured on
the recorded pilots: pilot density (the NR case) and observation aperture.

    python pilot_density.py ../data/captures/capture_icc_WARM_<stamp>.mat --json ../paper/figures/fig_density_numbers.json

Needs a capture recorded with ICC_LOG_PILOTS=1 (fields Rp_DL / Rp_UL, shape
R x NPIL x NSYM).  The recorded comb is decimated in FREQUENCY (keep every F-th
pilot), in TIME (keep every T-th symbol) and in APERTURE (keep the first M'
symbols); the per-burst pilot-slope estimator of acquisition/dsp.py is re-run on the
subset, the coarse (preamble) part is kept as logged, and the raw half sum is
scored on the held-out static window.  No tare and no drift correction: at the
native 78 ms gap both are below the floor, and this script is about precision.

THE CLOSED FORM (white noise only).  After coherent combining over Np pilots the
per-symbol phase error has variance 1/(2 rho Np), rho = per-pilot SNR.  A least
squares slope over the pilot-symbol instants t_m has variance
sigma_phi^2 / S with S = sum (t_m - tbar)^2, so per direction

    var(nu) = 1 / (2 rho Np (2 pi)^2 S),   S = Ts^2 M (M^2-1)/12 for M uniform symbols,

and the half sum has var = (var_D + var_U)/4.  rho is the pilot-EVM SNR logged
per burst (one value per burst, as in the paper).  Checked 2026-09-28: removing
each symbol's common phase first changes rho by 0.1 dB, so the EVM does not
absorb the oscillator wander.  The band-edge subcarriers are ~4 dB weaker than
the centre, and a residual-based per-subcarrier SNR overstates their noise, so
subsets made only of edge pilots sit slightly below this bound.  With centre
pilots only, the excess is 0.16-0.18 Hz for 2, 4 and 8 pilots: it is common to
all subcarriers, as the model assumes.

THE MODEL.  observed^2 = white^2 + excess^2.  The excess (oscillator phase
wander) is common to all subcarriers, so it does not fall with Np, and it is
fitted here against the aperture as excess ~ M^-gamma.  gamma = 1/2 is white
frequency noise (a phase random walk); gamma = 3/2 would be white phase noise.

For a CFO phase-slope estimator an NR frame differs from this one only in the
symbol spacing and in which (subcarrier, symbol) cells carry a pilot, so comb-12
of the recorded comb (every 24th subcarrier) with a pilot on every symbol IS the
NR PT-RS case (K=2, L=1) on this hardware.

Writes a JSON file (grid, aperture, model) that make_numbers.py reads; the paper
figure is drawn by matlab/fig_density_m.m from the same capture.
"""
import sys, os, json, argparse
import numpy as np
from icc_common import load, heldout
from bidir_estimators import get_variant

F_LIST = [1, 2, 4, 8, 12, 24]      # frequency decimation of the recorded comb
T_LIST = [1, 2, 4]                 # time decimation (every T-th symbol)
M_LIST = [16, 24, 32, 48, 64, 96, 128]   # aperture: first M' symbols (plus the full NSYM)
GAMMA_FROM = 32                    # fit the excess only where the unwrap is reliable


def _wls(phi, t, w):
    W = w.sum(); tb = float((w * t).sum() / W); pb = float((w * phi).sum() / W)
    den = float((w * (t - tb) ** 2).sum())
    s = float((w * (t - tb) * (phi - pb)).sum() / den) if den > 0 else np.nan
    return s, pb, tb


def slope_hz(Rp, tsym):
    """The acquisition/dsp.py two-pass estimator on an (npil, nsym) pilot matrix -> residual CFO [Hz]."""
    npil, nsym = Rp.shape
    T = np.arange(nsym) * tsym
    phi0 = np.unwrap(np.angle(np.sum(Rp * np.conj(Rp[:, [0]]), axis=0)))
    s0, _, _ = _wls(phi0, T, np.ones(nsym))
    H = (Rp * np.exp(-1j * s0 * T)[None, :]).mean(axis=1)
    z = np.sum(Rp * np.conj(H)[:, None], axis=0)
    phi = np.unwrap(np.angle(z)); w = np.abs(z) ** 2
    s1, p1, t1 = _wls(phi, T, w)
    res = phi - (p1 + s1 * (T - t1))
    thr = max(0.5 * np.pi, 3.0 * 1.4826 * np.median(np.abs(res - np.median(res))))
    keep = np.abs(res) <= thr
    if keep.sum() < min(8, nsym):
        keep[:] = True
    s, _, _ = _wls(phi[keep], T[keep], w[keep])
    return s / (2 * np.pi)


def rho_clean(Rp):
    """Per-SUBCARRIER pilot SNR with the per-symbol common phase removed [linear], (npil,).

    Rp (npil, nsym).  Each symbol is derotated by the phase of its coherent pilot
    sum (this strips CFO, drift and oscillator wander alike), the channel is the
    mean over symbols, and the residual is white noise.  Kept per subcarrier
    because the band edges are weaker than the centre: a sparse subset has its
    own SNR, not the average one.  Removing one phase per symbol uses one of the
    npil degrees of freedom, hence npil/(npil-1)."""
    npil, nsym = Rp.shape
    H0 = Rp.mean(axis=1)
    z = np.sum(Rp * np.conj(H0)[:, None], axis=0)
    Rd = Rp * np.exp(-1j * np.angle(z))[None, :]
    H = Rd.mean(axis=1)
    nv = np.mean(np.abs(Rd - H[:, None]) ** 2, axis=1) * npil / (npil - 1) * nsym / (nsym - 1)
    return np.abs(H) ** 2 / nv


def white_bound(rho_d, rho_u, fsel, t_inst):
    """Closed-form white-noise std of the half sum [Hz], rms over exchanges.

    rho_d, rho_u (n_exch, npil) per-subcarrier SNR; coherent (MRC) combining over
    the kept pilots gives a per-symbol phase variance 1/(2 sum_k rho_k), which is
    1/(2 Np rho) for equal SNR."""
    S = float(np.sum((t_inst - t_inst.mean()) ** 2))
    k = 1.0 / (2 * (2 * np.pi) ** 2 * S)
    vd = k / rho_d[:, fsel].sum(axis=1); vu = k / rho_u[:, fsel].sum(axis=1)
    return float(np.sqrt(np.mean((vd + vu) / 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture'); ap.add_argument('--json', default=None)
    a = ap.parse_args()
    d = load(a.capture)
    if 'Rp_DL' not in d:
        sys.exit("this capture has no recorded pilots -- record with ICC_LOG_PILOTS=1")
    A, B = np.asarray(d['Rp_DL']), np.asarray(d['Rp_UL'])
    R, npil, nsym = A.shape
    tsym = float(d['TSYM']); hz = float(d['HZ_PER_MPS'])
    co_d = np.asarray(get_variant(d, 'cfo_A2B_coarse'), float)
    co_u = np.asarray(get_variant(d, 'cfo_B2A_coarse'), float)
    ho = heldout(d); idx = np.where(ho)[0]
    evm_d = 10 ** (np.asarray(get_variant(d, 'snr_at_B'), float)[idx] / 10)
    evm_u = 10 ** (np.asarray(get_variant(d, 'snr_at_A'), float)[idx] / 10)
    rho_d = np.repeat(evm_d[:, None], npil, axis=1)          # one SNR per burst, all pilots
    rho_u = np.repeat(evm_u[:, None], npil, axis=1)
    evm = 10 * np.log10(np.r_[evm_d, evm_u])
    rc = np.array([rho_clean(A[i]) for i in idx[::10]])      # diagnostic only
    print(f"  per-subcarrier SNR (common phase removed): edge {10*np.log10(np.median(rc[:, [0, -1]])):.1f} dB, "
          f"centre {10*np.log10(np.median(rc[:, 12:18])):.1f} dB")
    print(f"{os.path.basename(a.capture)}: {A.shape}, {idx.size} held-out exchanges, "
          f"pilot EVM SNR {np.median(evm):.1f} dB")

    def run(fsel, tsel):
        fd = np.full(idx.size, np.nan)
        sp = (tsel[1] - tsel[0]) * tsym
        for j, i in enumerate(idx):
            a_, b_ = A[i][fsel][:, tsel], B[i][fsel][:, tsel]
            if np.all(np.isfinite(a_)) and np.all(np.isfinite(b_)):
                fd[j] = (co_d[i] + slope_hz(a_, sp) + co_u[i] + slope_hz(b_, sp)) / 2.0
        x = fd[np.isfinite(fd)]
        wb = white_bound(rho_d, rho_u, fsel, tsel * tsym)
        return dict(n_pilots=int(len(fsel)), n_symbols=int(len(tsel)),
                    aperture_ms=float((tsel[-1] - tsel[0] + 1) * tsym * 1e3),
                    overhead=len(fsel) * len(tsel) / (npil * nsym),
                    std_hz=float(np.std(x)), mean_hz=float(np.mean(x)), rmse_hz=float(np.sqrt(np.mean(x ** 2))),
                    white_hz=wb, std=float(np.std(x)) / hz, rmse=float(np.sqrt(np.mean(x ** 2))) / hz,
                    white=wb / hz, n=int(x.size))

    grid = []
    for T in T_LIST:
        for F in F_LIST:
            r = run(np.arange(0, npil, F), np.arange(0, nsym, T)) | dict(F=F, T=T)
            grid.append(r)
            print(f"  comb x{F:2d} every {T} sym  Np={r['n_pilots']:2d} Nt={r['n_symbols']:3d}  "
                  f"std {r['std']:.4f}  white {r['white']:.4f} m/s  ratio {r['std_hz']/r['white_hz']:.2f}  mean {r['mean_hz']:+.2f} Hz")
    aper = []
    for Mp in M_LIST + [nsym]:
        r = run(np.arange(npil), np.arange(Mp)) | dict(M=int(Mp))
        aper.append(r)
        print(f"  aperture M={Mp:3d} ({r['aperture_ms']:.2f} ms)  std {r['std']:.4f}  white {r['white']:.4f} m/s")

    # --- the model: observed^2 = white^2 + excess^2, excess ~ M^-gamma -----------
    full = next(r for r in grid if r['F'] == 1 and r['T'] == 1)
    ex = lambda r: np.sqrt(max(r['std_hz'] ** 2 - r['white_hz'] ** 2, 0.0))
    pts = [(r['M'], ex(r)) for r in aper if r['M'] >= GAMMA_FROM and ex(r) > 0]
    gamma = -float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0])
    ex_full = ex(full)
    for r in grid:
        r['model_hz'] = float(np.sqrt(r['white_hz'] ** 2 + ex_full ** 2)); r['model'] = r['model_hz'] / hz
    for r in aper:
        e = ex_full * (r['M'] / nsym) ** (-gamma)
        r['model_hz'] = float(np.sqrt(r['white_hz'] ** 2 + e ** 2)); r['model'] = r['model_hz'] / hz
    # excess measured on the Np decimations (should be flat in Np)
    ex_np = [ex(r) for r in grid if r['T'] == 1 and r['n_pilots'] >= 8]
    # pilot count at which white noise equals the excess
    np_star = full['n_pilots'] * (full['white_hz'] / ex_full) ** 2
    model = dict(white_hz=full['white_hz'], obs_hz=full['std_hz'], excess_hz=ex_full, gamma=gamma,
                 excess_np_min=float(min(ex_np)), excess_np_max=float(max(ex_np)), np_star=float(np_star),
                 hz_per_mps=hz, rho_db=float(10 * np.log10(np.median(np.r_[rho_d, rho_u]))), evm_db=float(np.median(evm)),
                 tsym=tsym, npil=int(npil), nsym=int(nsym))
    print(f"  model: white {full['white_hz']:.3f} Hz, observed {full['std_hz']:.3f} Hz, excess {ex_full:.3f} Hz "
          f"(Np>=8: {min(ex_np):.3f}-{max(ex_np):.3f}), gamma {gamma:.2f}, Np* {np_star:.1f}")
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(dict(grid=grid, aperture=aper, model=model), fh, indent=1)
        print(f"wrote {a.json}")


if __name__ == '__main__':
    main()
