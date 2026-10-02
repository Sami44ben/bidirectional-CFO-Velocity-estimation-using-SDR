"""
make_numbers.py -- regenerate ../paper/results_numbers.tex from captures.

    python make_numbers.py --warm ../data/captures/capture_icc_WARM_<stamp>.mat \
                           --sep  "../data/captures/capture_icc_SEP_<stamp>_*.mat" \
                           --gap-warm "../data/captures/capture_icc_GAP_<stampW>_*.mat" \
                           --gap-cold "../data/captures/capture_icc_GAP_<stampC>_*.mat" \
                           --move ../data/captures/capture_icc_MOVE_REF.mat \
                           [--fan-json  ../paper/figures/fig_fan_numbers.json] \
                           [--motion-json ../paper/figures/fig_motion_ref_numbers.json] \
                           [--density-json ../paper/figures/fig_density_numbers.json] \
                           [--sim "../data/emulated/OFFLINE_capture_icc_SEP_*.mat"] [--out PATH] [--dry]

Every macro the manuscript uses is written.  A macro whose source is missing is
written as \\tbd (a red ??), never as a stale value.  The previous file is kept
as results_numbers.prev.tex.  Metric definitions come from icc_common.py (the
same ones analyze_icc.py prints), so the paper and the console agree.
"""
import os, sys, glob, json, argparse, datetime
import numpy as np
from icc_common import load, heldout, masks, metrics, summary
from bidir_estimators import get_variant, DRIFT_WINDOW

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', 'paper', 'results_numbers.tex'))
TBD = None                      # sentinel -> \tbd
M = {}                          # macro -> value (number, string, or TBD)
SRC = {}                        # macro group -> provenance line
REFS = []                       # zero-injection reference runs from every session (tare stability)


def put(name, val, fmt=None, signed=False):
    """Store a macro.  Floats are formatted; negatives get \\ensuremath{-x} as before."""
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        M[name] = TBD; return
    if isinstance(val, str):
        M[name] = val; return
    s = format(val, fmt) if fmt else (f'{val:g}' if isinstance(val, float) else str(val))
    if signed and val > 0:
        s = '+' + s
    if s.startswith('-'):
        s = '\\ensuremath{' + s + '}'
    M[name] = s


def col(d, name):
    v = get_variant(d, name)
    return None if v is None else np.asarray(v, float)


def slope_hz_per_s(t, f):
    ok = np.isfinite(t) & np.isfinite(f)
    return float(np.polyfit(t[ok], f[ok], 1)[0]) if ok.sum() > 3 else np.nan


def files(pattern):
    if not pattern:
        return []
    fs = sorted(glob.glob(pattern)) or sorted(glob.glob(os.path.join(HERE, pattern)))
    return fs


# --------------------------------------------------------------------------- static floor / acquisition
def do_warm(path):
    d = load(path); s = summary(d)
    hz = s['hz_per_mps']
    ho = heldout(d); _, _, cal = masks(d)
    t = col(d, 't_A2B_rel'); gap = col(d, 'gap_s')
    upd = np.diff(t)
    # acquisition
    put('acqRate', s['rate_hz'], '.2f'); put('acqGap', s['gap_ms_med'], '.0f'); put('acqGapMax', s['gap_ms_max'], '.0f')
    put('acqUpdate', s['upd_ms_med'], '.0f'); put('acqUpdateMax', s['upd_ms_max'], '.0f')
    put('apertureMs', s['burst_ms'], '.2f')
    payload = s['nsym'] * (s['fft'] + s['fft'] // 8); frame = payload + 2 * int(round(0.1 * payload)) + (2 * s['fft'] + s['fft'] // 8)
    rxbuf = 1 << int(np.ceil(np.log2(3 * frame)))
    put('rxBuf', int(rxbuf)); put('rxBufMs', rxbuf / (s['fs_msps'] * 1e6) * 1e3, '.1f')
    pauses = upd > 3 * np.median(upd)
    put('stallNewN', int(pauses.sum())); put('stallNewMax', s['upd_ms_max'] / 1e3, '.2f')
    # config macros (so the tex never hard-codes numerology)
    put('cfgFc', s['fc_ghz'], '.1f'); put('cfgFs', s['fs_msps'], '.2f'); put('cfgFft', int(s['fft']))
    put('cfgDf', s['df_khz'], '.0f'); put('cfgNact', int(s['nact'])); put('cfgNp', int(s['npil']))
    put('cfgM', int(s['nsym'])); put('cfgTs', s['tsym_us'], '.2f'); put('cfgCp', s['cp_us'], '.2f')
    put('cfgBw', s['bw_mhz'], '.1f'); put('cfgTx', int(s['tx_gain'])); put('cfgRx', int(s['rx_gain']))
    put('cfgGuard', int(round(0.1 * payload)))
    put('hzPerMps', hz, '.2f')
    # floor
    put('floorTotal', s['n']); put('floorN', int(ho.sum())); put('floorDur', s['dur_s'] / 60, '.1f')
    v_plain = col(d, 'v_bidir_plain'); v_ta = col(d, 'v_bidir_tinterp'); v_dc = col(d, 'v_bidir_driftcomp')
    v_1 = col(d, 'v_oneway_nocorr'); v_afc = col(d, 'v_oneway_afc'); v_dis = col(d, 'v_node_disagree')
    mp = metrics(v_plain[ho]); put('floorHalf', mp['rmse'], '.3f'); put('floorHalfBias', mp['bias'], '.3f'); put('floorHalfPnf', mp['p95'], '.3f')
    # beta_+ : the constant of the half sum, measured as its mean over the held-out
    # static window (true velocity zero).  No tare is applied anywhere any more.
    const_hz = mp['bias'] * hz
    put('betaMps', mp['bias'], '.3f'); put('betaHz', const_hz, '.2f')
    put('floorHalfHz', float(np.nanstd(v_plain[ho])) * hz, '.3f')
    # residual directional bias after removing the drift term on every exchange
    # (mean of the drift-compensated half sum; the raw mean still carries -r*Delta/2)
    if v_dc is not None:
        put('betaAdjHz', float(np.nanmean(v_dc[ho])) * hz, '.2f'); put('betaAdjMps', float(np.nanmean(v_dc[ho])), '.3f')
    put('gapJitterMs', float(np.nanstd(gap[ho])) * 1e3, '.0f')
    for name, vv in (('floorTa', v_ta), ('floorDcomp', v_dc), ('floorOneway', v_1), ('floorOnewayAfc', v_afc), ('floorNodeRms', v_dis)):
        put(name, metrics(vv[ho])['rmse'] if vv is not None else np.nan, '.0f' if name == 'floorOneway' else ('.1f' if name == 'floorOnewayAfc' else '.3f'))
    if v_ta is not None: put('floorTaPnf', metrics(v_ta[ho])['p95'], '.3f')
    if v_dc is not None: put('floorDcompPnf', metrics(v_dc[ho])['p95'], '.3f')
    if v_afc is not None: put('floorOnewayAfcPnf', metrics(v_afc[ho])['p95'], '.1f')
    f_lo = col(d, 'f_lo_offset'); fd = col(d, 'fd_plain')
    put('floorDrift', slope_hz_per_s(t[ho], f_lo[ho]), '.3f')
    D = col(d, 'drift_hz_per_s')
    if D is not None and np.isfinite(D).any():
        put('floorDriftMin', np.nanmin(D[ho]), '.1f'); put('floorDriftMax', np.nanmax(D[ho]), '.1f', signed=True)
    se_obs = float(np.nanstd(fd[ho])); se_pred = float(np.sqrt(np.nanmean(col(d, 'se_at_B')[ho] ** 2 + col(d, 'se_at_A')[ho] ** 2)) / 2)
    put('floorSeObs', se_obs, '.3f'); put('floorSePred', se_pred, '.3f'); put('floorSeOptimism', se_obs / se_pred if se_pred > 0 else np.nan, '.1f')
    put('floorSnrDL', s['snr_dl'], '.1f'); put('floorSnrUL', s['snr_ul'], '.1f'); put('floorStalls', int(pauses.sum()))
    put('gapWindowExch', int(DRIFT_WINDOW)); put('gapWindowS', DRIFT_WINDOW * s['upd_ms_med'] / 1e3, '.1f')
    # --- acceleration limits (closed form, evaluated at this configuration) ---
    # each leak set equal to the static floor sigma_v:
    #   LO estimate   : a Delta/2        -> a* = 2 sigma_v / Delta
    #   raw half sum  : j Delta^2/8      -> j* = 8 sigma_v / Delta^2
    #   drift comp.   : +j Delta^2/8 (midpoint) - j Delta^2/4 (through r_hat) = -j Delta^2/8 -> same j* as raw
    #   alignment     : j T_u^2/16       -> j* = 16 sigma_v / T_u^2
    #   within burst  : unwrap safe while pi (f_c/c) a (T_a/2)^2 < pi/2 -> a < 2c/(f_c T_a^2)
    sv = mp['rmse']; De = s['gap_ms_med'] / 1e3; Tu = s['upd_ms_med'] / 1e3; Ta = s['burst_ms'] / 1e3
    put('accLoMax', 2 * sv / De, '.1f'); put('jerkRawMax', 8 * sv / De ** 2, '.0f')
    put('jerkDcMax', 8 * sv / De ** 2, '.0f'); put('jerkAlignMax', 16 * sv / Tu ** 2, '.0f')
    put('accBurstMax', 2.0 / hz / Ta ** 2 / 1e3, '.0f')          # 2c/(f_c Ta^2) = 2/(hz Ta^2), in km/s^2
    SRC['warm'] = f"{os.path.basename(path)}: {s['n']} exch, {s['dur_s']/60:.1f} min, floor {mp['rmse']:.3f} m/s"
    return dict(beta_hz=const_hz, hz=hz, sigma_hz=float(np.nanstd(v_plain[ho])) * hz)


# --------------------------------------------------------------------------- separation matrix
def do_sep(paths, hz, sim=False):
    rows = []
    for p in paths:
        d = load(p)
        fd_inj = float(d.get('FD_INJECT_HW', d.get('FD_INJECT', 0.0))); lo_inj = float(d.get('LO_OFFSET_B_HZ', d.get('LO_OFFSET', 0.0)))
        fd = col(d, 'fd_plain'); flo = col(d, 'f_lo_offset'); vp = col(d, 'v_bidir_plain')
        ok = np.isfinite(fd) & np.isfinite(flo)
        fdc = col(d, 'fd_driftcomp')
        rows.append(dict(fd_inj=fd_inj, lo_inj=lo_inj, fd=float(fd[ok].mean()), flo=float(flo[ok].mean()),
                         fdc=float(np.nanmean(fdc[ok])) if fdc is not None else np.nan,
                         sd_v=float(np.nanstd(vp[ok])), n=int(ok.sum()), t0=float(d.get('T0_EPOCH', 0.0))))
    rows.sort(key=lambda r: r['t0'])
    X = np.array([[r['fd_inj'], r['lo_inj'], 1.0] for r in rows]); yf = np.array([r['fd'] for r in rows]); yl = np.array([r['flo'] for r in rows])
    def ols(X, y):
        b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ b; dof = max(len(y) - X.shape[1], 1); s2 = float(r @ r) / dof
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        return b, se, r
    bf, sef, rf = ols(X, yf); bl, sel_, rl = ols(X, yl)
    pre = 'sim' if sim else 'sep'
    if sim:
        put('simGainD', bf[0], '.4f'); put('simGainLO', abs(bl[1]), '.4f')
        put('simLeakMps', abs(bf[1] * bl[2]) / hz, '.4f')
        SRC['sim'] = f"{len(rows)} emulated-channel runs"
        return
    put('sepGainD', bf[0], '.4f'); put('sepGainDse', sef[0], '.4f')
    put('sepGainLO', abs(bl[1]), '.3f'); put('sepGainLOse', sel_[1], '.3f')
    lf = bf[1]; put('sepLeakD', f'{lf:.0e}'.replace('e-0', '\\times10^{-').replace('e-', '\\times10^{-') + '}' if abs(lf) < 1e-2 else f'{lf:.4f}')
    put('sepLeakLO', bl[0], '.2f'); put('sepLeakLOse', sel_[0], '.2f')
    fdmax = max(abs(r['fd_inj']) for r in rows); put('sepLeakLOHz', abs(bl[0] * fdmax), '.0f')
    put('sepNatLO', abs(bl[2]), '.0f'); put('sepLeakMps', abs(lf * bl[2]) / hz, '.4f')
    put('sepConst', bf[2], '.2f'); put('sepFloorMps', float(np.mean([r['sd_v'] for r in rows])), '.3f')
    # 95 % interval of the leakage (t quantile, one point per run) and the worst leak in m/s
    from scipy import stats
    tq = stats.t.ppf(0.975, max(len(rows) - 3, 1)); lim = abs(lf) + tq * sef[1]
    e = int(np.floor(np.log10(lim))); m_ = lim / 10 ** e
    put('sepLeakCI', f'{np.ceil(m_):.0f}' + r'\times10^{' + str(e) + '}'); put('sepLeakCIMps', lim * abs(bl[2]) / hz, '.4f')
    put('betaSepAdjHz', float(np.nanmean([r['fdc'] - r['fd_inj'] for r in rows])), '.2f')
    refs = [r for r in rows if r['fd_inj'] == 0 and r['lo_inj'] == 0]
    REFS.extend(dict(t0=r['t0'], fd=r['fd'], flo=r['flo']) for r in refs)
    put('tareSepHz', bf[2], '.2f')
    order = np.arange(len(rows)); put('sepOrderCorr', float(np.corrcoef(order, rf)[0, 1]) if len(rows) > 2 else np.nan, '.2f')
    conds = {(r['fd_inj'], r['lo_inj']) for r in rows}
    put('sepConds', len(conds)); put('sepRefs', max(len(refs) - 1, 0)); put('sepExch', int(np.median([r['n'] for r in rows])))
    put('sepVmax', fdmax / hz, '.0f'); put('sepLOstep', max(abs(r['lo_inj']) for r in rows), '.0f')
    SRC['sep'] = f"{len(rows)} runs, g_f={bf[0]:.4f}, leak {lf:.1e}, o_f={bf[2]:+.2f} Hz"


# --------------------------------------------------------------------------- gap sweeps
def gap_rows(paths, hz):
    """One row per gap run: measured RMSE of the three estimators and their closed-form
    predictions from the SAME run's drift (no tare).  sigma0 is the white part of the
    half-sum scatter (first differences / sqrt 2); r(t) and eps_ddot come from a
    quadratic fit of the half difference over the run.  W = DRIFT_WINDOW (trailing).
        raw     : sigma0^2 + mean (r Delta/2)^2
        dcomp   : sigma0^2 + (eps_ddot (W-1) T_u Delta/4)^2 + (sigma0 Delta / (T_u sqrt(W(W^2-1)/3)))^2
        aligned : sigma0^2 + (eps_ddot T_u^2/16)^2"""
    rows = []
    for p in paths:
        d = load(p)
        t = col(d, 't_A2B_rel'); gap = col(d, 'gap_s'); fd = col(d, 'fd_plain'); flo = col(d, 'f_lo_offset')
        fdc = col(d, 'fd_driftcomp'); fta = col(d, 'fd_tinterp')
        ok = np.isfinite(fd) & np.isfinite(flo)
        m = ok & np.isfinite(fdc) & np.isfinite(fta)              # common support of all three
        D = slope_hz_per_s(t[ok], flo[ok]); g = float(np.nanmedian(gap)); Tu = float(np.nanmedian(np.diff(t)))
        c = np.polyfit(t[ok], flo[ok], 2); r_t = np.polyval(np.polyder(c), t); edd = 2 * c[0]
        s0 = float(np.nanstd(np.diff(fd[ok])) / np.sqrt(2))
        W = DRIFT_WINDOW
        p_raw = np.sqrt(s0 ** 2 + np.mean((r_t[m] * gap[m] / 2) ** 2))
        p_dc = np.sqrt(s0 ** 2 + (edd * (W - 1) * Tu * g / 4) ** 2 + (s0 * g / (Tu * np.sqrt(W * (W ** 2 - 1) / 3))) ** 2)
        p_al = np.sqrt(s0 ** 2 + (edd * Tu ** 2 / 16) ** 2)
        rms = lambda x: float(np.sqrt(np.mean(x[np.isfinite(x)] ** 2)))
        r = dict(gap=g, D=D, pred=-D * g / 2, meas=float(np.nanmean(fd[ok])),
                 resid_dc=float(np.nanmean(fdc[m])), Tu=Tu, edd=float(edd), s0=s0 / hz,
                 rmse_plain=rms(fd[m] / hz), rmse_dc=rms(fdc[m] / hz), rmse_ta=rms(fta[m] / hz),
                 pred_plain=p_raw / hz, pred_dc=p_dc / hz, pred_ta=p_al / hz,
                 n=int(ok.sum()), t0=float(d.get('T0_EPOCH', 0.0)), extra=float(d.get('GAP_S', 0.0)))
        rows.append(r)
    rows.sort(key=lambda r: r['t0'])
    return rows


def pick(rows, target_extra):
    """The run whose EXTRA dwell is closest to target (seconds); averaged if repeated."""
    c = [r for r in rows if abs(r['extra'] - target_extra) < 0.05]
    if not c:
        return None
    out = {k: float(np.mean([r[k] for r in c])) for k in c[0] if isinstance(c[0][k], float)}
    out['n'] = int(np.sum([r['n'] for r in c])); return out


def do_gap(paths_warm, paths_cold, hz, sigma_hz):
    allrows = []; rw = []; rc = []
    if paths_warm:
        rw = gap_rows(paths_warm, hz); allrows += rw
        extras = sorted({r['extra'] for r in rw})
        REFS.extend(dict(t0=r['t0'], fd=r['meas'], flo=np.nan) for r in rw if r['extra'] == 0)
        put('gapPts', len(extras)); put('gapRefs', sum(1 for r in rw if r['extra'] == 0) - 1)
        put('gapExch', int(np.median([r['n'] for r in rw])))
        put('gapMin', min(r['gap'] for r in rw), '.2f'); put('gapMax', max(r['gap'] for r in rw), '.1f')
        nat, one, two, four, mid = pick(rw, 0.0), pick(rw, 1.0), pick(rw, 2.0), pick(rw, 4.0), pick(rw, 0.5)
        for tag, r in (('Native', nat), ('One', one), ('Two', two), ('Four', four), ('Mid', mid)):
            if r is None: continue
            put(f'gap{tag}', r['gap'], '.2f')
            put(f'gapDrift{tag}', r['D'], '.2f', signed=True); put(f'gapPred{tag}', r['pred'], '.2f', signed=True)
            put(f'gapMeas{tag}', r['meas'], '.2f', signed=True)
            put(f'gapRmsePlain{tag}', r['rmse_plain'], '.3f'); put(f'gapRmseDcomp{tag}', r['rmse_dc'], '.3f'); put(f'gapRmseTa{tag}', r['rmse_ta'], '.3f')
        if four: put('gapFailFour', abs(four['meas'] - four['pred']), '.2f')
        short = [r['rmse_plain'] for r in rw if r['gap'] <= 0.7]
        put('gapRmsePlainShort', f'{min(short):.3f}--{max(short):.3f}' if short else None)
        SRC['gap_warm'] = f"{len(rw)} runs, native gap {rw[0]['gap']*1e3:.0f} ms"
    if paths_cold:
        rc = gap_rows(paths_cold, hz); allrows += rc
        put('cgapDriftStart', rc[0]['D'], '.1f'); put('cgapDriftEnd', abs(rc[-1]['D']), '.1f')
        nat, mid, four = pick(rc, 0.0), pick(rc, 0.5), pick(rc, 4.0)
        first_nat = next((r for r in rc if r['extra'] == 0), None)
        if first_nat:
            put('cgapBiasNative', first_nat['meas'], '.2f', signed=True); put('cgapPredNative', first_nat['pred'], '.2f', signed=True)
            put('cgapResidNative', first_nat['resid_dc'], '.2f', signed=True); put('betaColdShift', first_nat['resid_dc'], '.1f', signed=True)
            put('cgapRmsePlainNative', first_nat['rmse_plain'], '.3f'); put('cgapRmseDcompNative', first_nat['rmse_dc'], '.3f')
            put('cgapPredNativeRmse', first_nat['pred_plain'], '.3f')
        if mid: put('cgapRmsePlainMid', mid['rmse_plain'], '.3f'); put('cgapRmseDcompMid', mid['rmse_dc'], '.3f'); put('cgapMid', mid['gap'], '.2f')
        if four: put('cgapRmsePlainFour', four['rmse_plain'], '.3f'); put('cgapRmseDcompFour', four['rmse_dc'], '.3f')
        fit = [r for r in rc if r['gap'] <= 2.5]
        if len(fit) > 2:
            b = np.polyfit([r['pred'] for r in fit], [r['meas'] for r in fit], 1)
            res = np.array([r['meas'] - np.polyval(b, r['pred']) for r in fit])
            put('cgapSlope', b[0], '.2f'); put('cgapIntercept', b[1], '.2f', signed=True); put('cgapPredErr', float(np.abs(res).max()), '.2f')
            put('cgapModelMax', max(r['gap'] for r in fit), '.1f')
        ta = [r['rmse_ta'] for r in rc if np.isfinite(r['rmse_ta'])]
        put('cgapRmseTa', float(np.mean(ta)) if ta else np.nan, '.3f')
        if four: put('cgapRmseTaFour', four['rmse_ta'], '.3f')
        put('cgapEddMax', max(abs(r['edd']) for r in rc), '.2f')
        SRC['gap_cold'] = f"{len(rc)} runs, D {rc[0]['D']:+.1f} -> {rc[-1]['D']:+.1f} Hz/s"
    if allrows:
        # --- closed form against measurement, every gap run ---
        for est in ('plain', 'dc', 'ta'):
            e = np.array([r[f'pred_{est}'] / r[f'rmse_{est}'] - 1 for r in allrows])
            put(f'gapModelMed{est.capitalize()}', 100 * float(np.median(np.abs(e))), '.0f')
            put(f'gapModelMax{est.capitalize()}', 100 * float(np.max(np.abs(e))), '.0f')
        put('gapRuns', len(allrows))
        # --- tolerance: gap at which the drift bias equals the floor, Delta* = 2 sigma/|r|;
        #     RMSE +10 % at sqrt(1.1^2-1) = 0.46 of that ---
        if rw:
            rwarm = float(np.median([abs(r['D']) for r in rw]))
            put('gapRWarm', rwarm, '.2f'); put('gapTolWarm', 2 * sigma_hz / rwarm, '.1f')
            put('gapTolWarmTen', 2 * sigma_hz * np.sqrt(0.21) / rwarm, '.1f')
        if rc:
            rcold = abs(rc[0]['D'])
            put('gapTolCold', 1e3 * 2 * sigma_hz / rcold, '.0f'); put('gapTolColdTen', 1e3 * 2 * sigma_hz * np.sqrt(0.21) / rcold, '.0f')
            edd = max(abs(r['edd']) for r in rc)
            put('gapTolDcCold', np.sqrt(4 * sigma_hz / (edd * (DRIFT_WINDOW - 1))), '.1f')
            put('gapTolTaCold', np.sqrt(16 * sigma_hz / edd), '.1f')
        with open(os.path.join(os.path.dirname(OUT), 'figures', 'fig_gap_numbers.json'), 'w') as fh:
            json.dump(dict(sigma_hz=sigma_hz, hz=hz, window=DRIFT_WINDOW,
                           warm=[r for r in rw], cold=[r for r in rc]), fh, indent=1)


# --------------------------------------------------------------------------- peer motion
def do_move(path, hz):
    d = load(path); s = summary(d)
    t = col(d, 't_A2B_rel'); gap = col(d, 'gap_s'); fd = col(d, 'fd_plain'); flo = col(d, 'f_lo_offset'); fdc = col(d, 'fd_driftcomp')
    st, dy, cal = masks(d); ho = heldout(d)
    put('movePairs', s['n']); put('moveGapMed', s['gap_ms_med'], '.0f')
    D = slope_hz_per_s(t[st], flo[st]); put('moveDrift', D, '.1f')
    put('moveBiasPred', -D * float(np.nanmedian(gap[ho])) / 2, '.2f', signed=True)
    put('moveBiasObs', float(np.nanmean(fd[ho] - fdc[ho])) if fdc is not None else np.nan, '.2f', signed=True)
    src = dy if dy.any() else np.ones_like(st)
    put('moveOutMin', float(np.nanmin(fd[src])), '.1f'); put('moveOutMax', float(np.nanmax(fd[src])), '.1f', signed=True)
    # qualitative motion description: bursts, speed range, net displacement, static bias
    v = col(d, 'v_bidir_driftcomp')                # drift compensated, no tare (the raw half sum carries -r*Delta/2)
    snrB = col(d, 'snr_at_B'); snrA = col(d, 'snr_at_A')
    ok = np.isfinite(v) & (snrB > 8) & (snrA > 8)
    vi = np.interp(t, t[ok], v[ok]); vs = np.convolve(vi, np.ones(5)/5, 'same')
    disp = np.concatenate([[0], np.cumsum(0.5*(vi[1:]+vi[:-1])*np.diff(t))])
    act = np.abs(vs) > 0.25; idx = np.where(dy if dy.any() else st)[0]
    runs = []; k = idx[0] if idx.size else 0
    while k < len(t):
        if act[k] and (dy[k] if dy.any() else True):
            j = k
            while j+1 < len(t) and (act[j+1] or (j+2 < len(t) and act[j+2])): j += 1
            if t[j]-t[k] > 0.8: runs.append((k, j))
            k = j+1
        else: k += 1
    if runs:
        sp = [abs(disp[b_]-disp[a_])/(t[b_]-t[a_]) for a_, b_ in runs]
        put('moveBursts', len(runs)); put('moveSpeedLo', min(sp), '.1f'); put('moveSpeedHi', max(sp), '.1f')
    put('moveNet', float(disp[-1]), '.1f', signed=True)
    put('moveStatBias', float(np.nanmean(v[st & ok & ~cal])) if (st & ok & ~cal).any() else np.nan, '.3f', signed=True)
    SRC['move'] = f"{os.path.basename(path)}: {s['n']} exch, D {D:+.1f} Hz/s"


def do_json(kind, path):
    if not path or not os.path.isfile(path):
        return
    j = json.load(open(path))
    if kind == 'fan':
        runs = j['runs']
        offs = [r for r in runs if r['label'].startswith('fan off')]
        ons = sorted([r for r in runs if r['label'].startswith('fan on')], key=lambda r: r.get('rpm') or 0)
        if ons:
            on = ons[0]                                   # slowest fan = the headline number
            put('fanOnRms', on['rms'], '.2f'); put('fanOnPnf', on['p95'], '.2f'); put('fanDisagree', on['disagree_rms'], '.2f')
            put('fanPairs', on['n']); put('fanRmsTaMps', on['rms'], '.2f'); put('fanPFiveTaMps', on['p95'], '.2f')
            put('fanRpm', on.get('rpm'), '.0f'); put('fanTip', on.get('tip_speed'), '.1f')
        if len(ons) > 1:
            fast = ons[-1]
            put('fanOnFastRms', fast['rms'], '.2f'); put('fanOnFastPnf', fast['p95'], '.2f')
            put('fanDisagreeFast', fast['disagree_rms'], '.2f'); put('fanRpmFast', fast.get('rpm'), '.0f')
            put('fanTipFast', fast.get('tip_speed'), '.1f')
        if offs:
            put('fanOffRms', float(np.mean([r['rms'] for r in offs])), '.3f')
            put('fanOffDisagree', float(np.mean([r['disagree_rms'] for r in offs])), '.3f')
        put('fanRadius', j.get('radius'), '.2f')
        SRC['fan'] = f"{path}"
    elif kind == 'motion':
        put('motionRuns', j['n_passes']); put('motionRmse', j['rmse'], '.3f'); put('motionPFive', j['p95'], '.3f')
        put('motionSlope', j['gain'], '.3f'); put('motionBias', j['bias'], '.3f', signed=True)
        put('motionRange', f"{abs(j['v_min']):.1f}--{abs(j['v_max']):.1f}"); put('refUncert', j['ref_unc'], '.3f')
        put('motionDisp', j['disp_rms'], '.2f'); put('motionL', j['L'], '.1f'); put('refInstrument', 'tape marks and 60 frame/s video')
        SRC['motion'] = f"{path}"
    elif kind == 'density':
        g = j['grid']; ap = j['aperture']; mo = j['model']; hz_ = mo['hz_per_mps']
        full = next(r for r in g if r['F'] == 1 and r['T'] == 1)
        k2 = next(r for r in g if r['F'] == 12 and r['T'] == 1)
        k2t4 = next(r for r in g if r['F'] == 12 and r['T'] == 4)
        put('densFullNp', full['n_pilots']); put('densFullRmse', full['std'], '.3f')
        put('densPtrsNp', k2['n_pilots']); put('densPtrsRmse', k2['std'], '.3f'); put('densPtrsRatio', k2['std'] / full['std'], '.1f')
        put('densPtrsModel', k2['model'], '.3f'); put('densPtrsWhite', k2['white'], '.3f')
        put('densPtrsOffset', abs(k2['mean_hz']), '.2f'); put('densPtrsOffsetMps', abs(k2['mean_hz']) / hz_, '.3f')
        put('densPtrsRmseTot', k2['rmse'], '.3f')
        put('densPtrsTfourRmse', k2t4['std'], '.3f'); put('densPtrsTfourOverhead', 100 * k2t4['overhead'], '.1f')
        put('precRhoDb', mo['rho_db'], '.0f'); put('precWhiteHz', mo['white_hz'], '.3f'); put('precWhiteMps', mo['white_hz'] / hz_, '.3f')
        put('precObsHz', mo['obs_hz'], '.3f'); put('precRatio', mo['obs_hz'] / mo['white_hz'], '.1f')
        put('precExcessHz', mo['excess_hz'], '.2f'); put('precExcessMin', mo['excess_np_min'], '.2f'); put('precExcessMax', mo['excess_np_max'], '.2f')
        put('precGamma', mo['gamma'], '.2f'); put('precNpStar', mo['np_star'], '.0f')
        big = [r for r in g if r['n_pilots'] >= 8]
        put('precModelErr', 100 * max(abs(r['model'] / r['std'] - 1) for r in big), '.0f')
        put('precModelErrPtrs', 100 * abs(k2['model'] / k2['std'] - 1), '.0f')
        put('aperModelErr', 100 * max(abs(r['model'] / r['std'] - 1) for r in ap if r['M'] >= 24), '.0f')
        a64 = next(r for r in ap if r['M'] == 64)
        put('aperHalfMs', a64['aperture_ms'], '.1f'); put('aperHalfStd', a64['std'], '.3f'); put('aperHalfWhite', a64['white'], '.3f')
        put('aperGain', a64['std'] / full['std'], '.1f'); put('aperGainWhite', a64['white'] / full['white'], '.1f')
        SRC['density'] = f"{path}"


# --------------------------------------------------------------------------- constants that are history, not measurement
CONST = {
    'seqSeed': '20260904',
    'stallOldN': '43', 'stallOldMax': '14.2', 'stallOldFile': '278', 'stallHitRate': '100', 'stallChance': '8', 'stallStep': '44',
}

USED = """acqGap acqGapMax acqRate apertureMs cgapBiasNative cgapDriftEnd cgapDriftStart cgapModelMax cgapPredErr cgapPredNative
cgapResidNative cgapRmseDcompFour cgapRmseDcompMid cgapRmseDcompNative cgapRmsePlainFour cgapRmsePlainMid cgapRmsePlainNative
cgapRmseTa cgapSlope fanDisagree fanOffRms fanOnPnf fanOnRms fanPFiveTaMps fanPairs fanRmsTaMps fanRpm floorDcomp floorDrift floorDur floorHalf floorHalfBias floorN floorNodeRms floorOneway floorOnewayAfc floorSeObs
floorSeOptimism floorSePred floorTa floorTotal gapDriftTwo gapExch gapMax gapMeasFour gapMeasOne gapMeasTwo gapPredFour gapPredOne
gapPredTwo gapRmseDcompOne gapRmseDcompTwo gapRmsePlainOne gapRmsePlainTwo gapRmseTaFour gapWindowExch motionPFive motionRange
motionRmse motionRuns motionSlope moveBiasObs moveBiasPred moveDrift moveOutMax moveOutMin movePairs refUncert rxBuf sepConds
sepConst sepExch sepFloorMps sepGainD sepGainDse sepGainLO sepGainLOse sepLOstep sepLOwander sepLeakD sepLeakLO sepLeakLOHz
sepLeakLOse sepLeakMps sepNatLO sepOrderCorr sepRefs sepVmax seqSeed simGainD simLeakMps stallNewMax stallOldMax stallOldN
betaHz betaMps betaColdShift tareSepHz tareSpanMin tareStabHz floorHalfHz
gapNative gapOne gapTwo gapFour gapMid cgapMid rxBufMs gapWindowS cfgFc cfgFs cfgFft cfgDf cfgNact cfgNp cfgM cfgTs cfgCp cfgBw
cfgTx cfgRx cfgGuard hzPerMps nRefs densFullNp densFullRmse densPtrsNp densPtrsRmse densPtrsRatio densPtrsTfourRmse densPtrsTfourOverhead
motionBias motionDisp motionL refInstrument fanTip moveBursts moveSpeedLo moveSpeedHi moveNet moveStatBias fanOnFastRms fanOnFastPnf fanDisagreeFast fanRpmFast fanTipFast fanOffDisagree fanRadius
accLoMax jerkRawMax jerkDcMax jerkAlignMax accBurstMax gapModelMedPlain gapModelMaxPlain gapModelMedDc gapModelMaxDc gapModelMedTa
gapModelMaxTa gapRuns gapRWarm gapTolWarm gapTolWarmTen gapTolCold gapTolColdTen gapTolDcCold gapTolTaCold cgapRmseTaFour cgapEddMax
densPtrsModel densPtrsWhite densPtrsOffset densPtrsOffsetMps precRhoDb precWhiteHz precWhiteMps precObsHz precRatio precExcessHz
betaAdjHz betaAdjMps gapJitterMs sepLeakCI sepLeakCIMps betaSepAdjHz cgapPredNativeRmse densPtrsRmseTot
precExcessMin precExcessMax precGamma precNpStar precModelErr precModelErrPtrs aperModelErr aperHalfMs aperHalfStd aperHalfWhite aperGain aperGainWhite floorTaPnf floorDcompPnf""".split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warm'); ap.add_argument('--sep'); ap.add_argument('--sim'); ap.add_argument('--gap-warm'); ap.add_argument('--gap-cold')
    ap.add_argument('--move'); ap.add_argument('--fan-json'); ap.add_argument('--motion-json'); ap.add_argument('--density-json')
    ap.add_argument('--out', default=OUT); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    for k, v in CONST.items():
        M[k] = v
    hz = 7.6667; sigma_hz = np.nan
    if a.warm:
        w = do_warm(a.warm); hz, sigma_hz = w['hz'], w['sigma_hz']
    if a.sep: do_sep(files(a.sep), hz)
    if a.sim: do_sep(files(a.sim), hz, sim=True)
    if a.gap_warm or a.gap_cold: do_gap(files(a.gap_warm), files(a.gap_cold), hz, sigma_hz)
    if a.move: do_move(a.move, hz)
    do_json('fan', a.fan_json); do_json('motion', a.motion_json); do_json('density', a.density_json)
    if len(REFS) > 1:
        fd_ref = np.array([r['fd'] for r in REFS]); flo_ref = np.array([r['flo'] for r in REFS]); t0 = np.array([r['t0'] for r in REFS])
        put('tareStabHz', float(np.std(fd_ref)), '.2f')
        put('tareSpanMin', float((t0.max() - t0.min()) / 60) if t0.max() > 0 else np.nan, '.0f')
        put('sepLOwander', float(np.nanstd(flo_ref)), '.0f')
        put('nRefs', int(len(REFS)))

    lines = [f"% results_numbers.tex -- GENERATED by Work/SDR/make_numbers.py on {datetime.datetime.now():%Y-%m-%d %H:%M}.",
             "% Do not edit by hand; re-run make_numbers.py.  \\tbd marks a value whose source run does not exist yet.",
             "\\newcommand{\\tbd}{{\\color{red}\\textbf{??}}}"]
    for k, v in SRC.items():
        lines.append(f"%   {k:9s} <- {v}")
    missing = []
    for name in USED:
        val = M.get(name, TBD)
        if val is TBD:
            missing.append(name); lines.append(f"\\newcommand{{\\{name}}}{{\\tbd}}")
        else:
            lines.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    for name in sorted(set(M) - set(USED)):                      # extras, harmless
        if M[name] is not TBD:
            lines.append(f"\\newcommand{{\\{name}}}{{{M[name]}}}")
    text = "\n".join(lines) + "\n"
    print(f"{len(USED)} macros: {len(USED)-len(missing)} filled, {len(missing)} \\tbd")
    if missing:
        print("  missing:", " ".join(missing))
    if a.dry:
        print(text); return
    if os.path.isfile(a.out):
        os.replace(a.out, a.out.replace('.tex', '.prev.tex'))
    open(a.out, 'w', encoding='utf-8').write(text)
    print(f"wrote {a.out}")


if __name__ == '__main__':
    main()
