"""
icc_common.py -- the few helpers every analysis script shares.

    load(path)            capture dict with every derived estimator added
    masks(d)              (static, dynamic, calibrating) boolean masks per exchange
    heldout(d)            static AND not calibrating  (the scored window)
    tare(d, key)          median of an estimator over the calibration window [m/s]
    metrics(err)          bias / RMSE / p95 / n_eff / block-bootstrap CI of an error vector
    summary(d)            one dict of the numerology + acquisition facts of a capture

metrics() and masks() are the definitions analyze_icc.py uses; they live here so
make_numbers.py, score_passes.py, fan_compare.py and pilot_density.py cannot
drift from it.
"""
import numpy as np
from scipy.io import loadmat
from bidir_estimators import add_estimators, get_variant

C_LIGHT = 3e8


def load(path):
    d = loadmat(path, squeeze_me=True)
    add_estimators(d)
    return d


def _f(d, k, default=np.nan):
    return float(d[k]) if k in d else default


def masks(d):
    """(static, dynamic, calibrating).  Captures without a phase column are treated
    as wholly static with the first CAL_S seconds (or 100 exchanges) calibrating."""
    t = np.asarray(get_variant(d, 't_A2B_rel'), float)
    R = t.size
    if 'phase' in d and _f(d, 'STATIC_S', 0.0) > 0.0:
        ph = np.atleast_1d(d['phase']).astype(float)
        if ph.size == R:
            cal_s = _f(d, 'CAL_S', 0.0)
            return (ph == 0), (ph == 1), (t < cal_s)
    cal = np.zeros(R, bool)
    cal_s = _f(d, 'CAL_S', 0.0)
    if cal_s > 0:
        cal[t < cal_s] = True
    else:
        cal[:min(100, R)] = True
    return np.ones(R, bool), np.zeros(R, bool), cal


def heldout(d):
    st, dy, cal = masks(d)
    return st & ~cal


def tare(d, key='v_bidir_plain'):
    """Median of an estimator over the calibration window.  Units of that estimator."""
    _, _, cal = masks(d)
    v = np.asarray(get_variant(d, key), float)
    v = v[cal & np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def metrics(err):
    """Signed bias, RMSE, std, tail percentiles, n_eff and a block-bootstrap CI."""
    e = np.asarray(err, float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return dict(n=0, bias=np.nan, rmse=np.nan, std=np.nan, med=np.nan,
                    p95=np.nan, p99=np.nan, n_eff=0, blk=1, ci_lo=np.nan, ci_hi=np.nan)
    a = np.abs(e)
    r1 = 0.0
    if e.size > 3:
        c = e - e.mean()
        dd = float(np.dot(c, c))
        r1 = float(np.dot(c[:-1], c[1:]) / dd) if dd > 0 else 0.0
        r1 = min(max(r1, 0.0), 0.99)
    n_eff = e.size * (1 - r1) / (1 + r1)
    L = max(1, int(round(e.size / max(n_eff, 1.0))))
    lo = hi = np.nan
    if e.size >= 8:
        rs, nb = np.random.default_rng(0), int(np.ceil(e.size / L))
        starts = rs.integers(0, e.size - L + 1, size=(400, nb))
        boot = np.array([np.sqrt((np.concatenate([e[t:t + L] for t in row])[:e.size] ** 2).mean())
                         for row in starts])
        lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=int(e.size), bias=float(e.mean()), rmse=float(np.sqrt((e ** 2).mean())),
                std=float(e.std()), med=float(np.median(a)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), n_eff=float(n_eff), blk=L,
                ci_lo=float(lo), ci_hi=float(hi))


def summary(d):
    """Numerology and acquisition facts of one capture, as plain floats."""
    t_dl = np.asarray(get_variant(d, 't_A2B_rel'), float)
    gap = np.asarray(get_variant(d, 'gap_s'), float)
    upd = np.diff(t_dl) if t_dl.size > 1 else np.array([np.nan])
    fs, fft, nsym, npil = _f(d, 'FS'), _f(d, 'FFTSIZE'), _f(d, 'NSYM'), _f(d, 'NPIL')
    tsym = _f(d, 'TSYM')
    cp = fft / 8.0
    return dict(
        fc_ghz=_f(d, 'FC') / 1e9, fs_msps=fs / 1e6, fft=fft, df_khz=_f(d, 'DF') / 1e3,
        nsym=nsym, npil=npil, ratio=_f(d, 'RATIO'), nact=npil * _f(d, 'RATIO'),
        tsym_us=tsym * 1e6, cp_us=cp / fs * 1e6, burst_ms=nsym * tsym * 1e3,
        bw_mhz=npil * _f(d, 'RATIO') * _f(d, 'DF') / 1e6,
        hz_per_mps=_f(d, 'HZ_PER_MPS'), n=int(t_dl.size),
        dur_s=float(t_dl[-1] - t_dl[0]) if t_dl.size > 1 else np.nan,
        rate_hz=float((t_dl.size - 1) / (t_dl[-1] - t_dl[0])) if t_dl.size > 1 else np.nan,
        gap_ms_med=float(np.nanmedian(gap) * 1e3), gap_ms_max=float(np.nanmax(gap) * 1e3),
        upd_ms_med=float(np.nanmedian(upd) * 1e3), upd_ms_max=float(np.nanmax(upd) * 1e3),
        snr_dl=float(np.nanmean(get_variant(d, 'snr_at_B'))),
        snr_ul=float(np.nanmean(get_variant(d, 'snr_at_A'))),
        tx_gain=_f(d, 'TX_GAIN'), rx_gain=_f(d, 'RX_GAIN'),
        gain_mode=str(d.get('GAIN_MODE', '')), tag=str(d.get('RUN_TAG', '')),
    )
