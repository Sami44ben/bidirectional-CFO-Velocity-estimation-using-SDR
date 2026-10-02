"""
analyze_icc.py -- OFFLINE analysis of the ICC campaign captures.

Reads capture_icc_*.mat written by acquisition/campaign.py and produces the numbers
and figures the paper needs.  Nothing here touches hardware.

  usage:  python analyze_icc.py                    # all capture_icc_*.mat here
          python analyze_icc.py capture_icc_B_static.mat ...

WHAT IT REPORTS (per run)
  * velocity metrics for the four estimators, against the KNOWN true velocity
    (FD_INJECT_HW/HZ_PER_MPS, which is 0 for every static and fan run):
       signed bias | RMSE | std | p95 | p99   <- reviewers ask for tails, not just RMSE
  * CFO cancellation ratio = how much oscillator contamination was removed (dB)
  * LO offset: mean, ppm, fitted drift rate (Hz/s), and the predicted SFO
  * estimator outage = exchanges the campaign loop attempted but could not use
    (found as GAPS in exch_id, since only successful exchanges get logged)
  * cycle-slip rate = rejected pilot symbols / total
  * DL-UL gap statistics (Exp. D's independent variable)

ACROSS RUNS
  * Exp. D table: RMSE of each estimator vs the measured DL-UL gap.
"""
import glob, os, sys, time
import numpy as np
from scipy.io import loadmat
from bidir_estimators import (add_estimators, get_variant, describe,
                              VARIANT_ORDER, LEGACY_ALIAS)

HERE       = os.path.dirname(os.path.abspath(__file__))
FILES      = ([f for a in sys.argv[1:] for f in (sorted(glob.glob(a)) or [a])]   # expand globs (PowerShell does not)
             or sorted(glob.glob(os.path.join(HERE, "..", "data", "captures", "capture_icc_*.mat"))))
SKIP_EXTRA = 0                       # drop this many EXTRA exchanges after calibration
MAKE_PLOTS = True
VARIANTS   = VARIANT_ORDER          # defined once, in bidir_estimators.py

def add_derived(d):
    """Every derived estimator comes from bidir_estimators, so the campaign
    script, this file and analyze_icc.m cannot drift apart.  Works on OLD
    captures too: everything is recomputed from the raw per-direction CFOs
    and timestamps that were always logged."""
    return add_estimators(d)


def phase_masks(d, R, t, n_cal):
    """Split a run into (static, dynamic, calibration) masks.

    WHY THIS EXISTS.  The fan and running captures are 'hold still, then move'.
    A true velocity of 0 is only valid while static, so scoring the moving stretch
    against 0 reports the motion itself as error -- which is exactly what the
    2026-09-07 figure did, with its 'true velocity +0.00 m/s' header and an error
    CDF that was really the distribution of how fast the operator ran.

    New captures carry a per-exchange 'phase' (0 static, 1 dynamic) and a CAL_S
    boundary, so nothing has to be guessed.  Old captures have neither and are
    treated as wholly static with the first N_CAL exchanges calibrating -- exactly
    what this script did before phases existed."""
    # STATIC_S = 0 means the run was never split: either an injected-Doppler
    # condition (the truth is known on EVERY exchange, so there is nothing to hold
    # out) or a plain continuous run.  Both must fall through to the legacy path --
    # marking such a run 'dynamic, no ground truth' would throw away a sweep
    # condition's entire scoring window.
    if 'phase' in d and float(d.get('STATIC_S', 0.0)) > 0.0:
        ph = np.atleast_1d(d['phase']).astype(float)
        if ph.size == R:
            cal_s = float(d['CAL_S']) if 'CAL_S' in d else 0.0
            return (ph == 0), (ph == 1), (t < cal_s)
    m = np.zeros(R, bool)
    m[:min(n_cal, R)] = True
    return np.ones(R, bool), np.zeros(R, bool), m


# --- the one helper that earns its keep (reused per estimator) ---
def metrics(err):
    """Signed bias, RMSE, std and tail percentiles of an error vector."""
    e = err[np.isfinite(err)]
    if e.size == 0:
        return dict(n=0, bias=np.nan, rmse=np.nan, std=np.nan, med=np.nan,
                    p95=np.nan, p99=np.nan, n_eff=0, blk=1, ci_lo=np.nan, ci_hi=np.nan)
    a = np.abs(e)
    # Effective sample size from the lag-1 autocorrelation.  Exchanges inside ONE run are
    # NOT independent (they share a warm-up drift trajectory and a thermal state), so the
    # raw n badly overstates confidence -- report n_eff and a block-bootstrap CI instead.
    r1 = 0.0
    if e.size > 3:
        c = e - e.mean()
        d = float(np.dot(c, c))
        r1 = float(np.dot(c[:-1], c[1:])/d) if d > 0 else 0.0
        r1 = min(max(r1, 0.0), 0.99)
    n_eff = e.size*(1 - r1)/(1 + r1)
    L     = max(1, int(round(e.size/max(n_eff, 1.0))))          # correlation block length
    lo = hi = np.nan
    if e.size >= 8:                                             # moving-block bootstrap
        rs, nb = np.random.default_rng(0), int(np.ceil(e.size/L))
        starts = rs.integers(0, e.size - L + 1, size=(400, nb))
        boot = np.array([np.sqrt((np.concatenate([e[t:t+L] for t in row])[:e.size]**2).mean())
                         for row in starts])
        lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n=e.size, bias=e.mean(), rmse=float(np.sqrt((e**2).mean())), std=e.std(),
                med=float(np.median(a)), p95=np.percentile(a, 95), p99=np.percentile(a, 99),
                n_eff=n_eff, blk=L, ci_lo=lo, ci_hi=hi)

if not FILES:
    sys.exit("No capture_icc_*.mat found. Run acquisition/campaign.py first.")

expD = []            # (gap_ms, tag, {variant: rmse})
print("\n" + "="*78)
print(f"ICC CAMPAIGN ANALYSIS -- {len(FILES)} run(s)")
print("="*78)

for fn in FILES:
    d   = add_derived(loadmat(fn, squeeze_me=True))
    tag = str(d.get('RUN_TAG', os.path.basename(fn)))
    fc, hz_mps = float(d['FC']), float(d['HZ_PER_MPS'])
    tsym, nsym = float(d['TSYM']), int(d['NSYM'])
    fs         = float(d['FS'])
    n_cal      = int(d['N_CAL']) if 'N_CAL' in d else 0
    fd_inj     = float(d['FD_INJECT_HW'])
    v_true     = fd_inj / hz_mps                      # the ground truth for this run

    exch = np.atleast_1d(d['exch_id']).astype(int)
    R    = exch.size
    _trel = (np.atleast_1d(d['t_dl_rel']).astype(float) if 't_dl_rel' in d
             else np.atleast_1d(d['t_dl']).astype(float) - float(np.atleast_1d(d['t_dl'])[0]))
    ph_static, ph_dyn, is_cal = phase_masks(d, R, _trel, n_cal)
    # Score ONLY on static, non-calibrating exchanges: that is the one stretch where
    # the true velocity is known (zero) and the bias was not fitted to the same data.
    keep = ph_static & ~is_cal
    if SKIP_EXTRA:
        keep[:min(int(np.sum(is_cal)) + SKIP_EXTRA, R)] = False
    if not keep.any():                     # nothing held out -- fall back, but say so
        keep = np.zeros(R, bool)
        keep[min(n_cal + SKIP_EXTRA, R):] = True
        if ph_static.all():
            pass
        else:
            print("    (no held-out static exchanges; scoring window fell back to post-N_CAL)")

    print(f"\n--- {tag}   ({os.path.basename(fn)})")
    note = d.get('NOTE', '')
    note = '' if not isinstance(note, str) else note.strip()
    if note:
        print(f"    note: {note}")
    print(f"    fc={fc/1e9:.2f} GHz  {hz_mps:.2f} Hz per m/s | aperture {nsym*tsym*1e3:.2f} ms"
          f" | exchanges {R} (scoring {keep.sum()})")
    if ph_dyn.any():
        print(f"    PHASES: {int(ph_static.sum())} static "
              f"({int(np.sum(is_cal))} of them calibrating, {int(keep.sum())} scored) "
              f"+ {int(ph_dyn.sum())} dynamic")
        print(f"    true velocity is {v_true:+.3f} m/s ONLY in the static phase; the "
              f"dynamic phase has no ground truth and is NOT scored as error.")
    else:
        print(f"    injected Doppler {fd_inj:+.2f} Hz  ->  TRUE velocity {v_true:+.3f} m/s")
    _tr = (np.atleast_1d(d['t_dl_rel']).astype(float) if 't_dl_rel' in d
           else np.atleast_1d(d['t_dl']).astype(float) - float(np.atleast_1d(d['t_dl'])[0]))
    _t0 = float(d['T0_EPOCH']) if 'T0_EPOCH' in d else np.nan
    _st = (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_t0))
           if np.isfinite(_t0) else 'unknown')
    print(f"    run window: t=+{_tr.min():.3f}s to +{_tr.max():.3f}s "
          f"({(_tr.max()-_tr.min())/60:.1f} min) | started {_st}")

    # ---------- velocity metrics per estimator ----------
    print(f"    {'estimator':<19}{'n':>5}{'neff':>7}{'bias':>10}{'RMSE':>10}"
          f"{'RMSE 95% CI':>20}{'med':>9}{'p95':>9}{'p99':>9}   [m/s]")
    got = {}
    for v in VARIANTS:
        x = get_variant(d, v)
        if x is None:
            continue
        err = x[keep] - v_true
        m   = metrics(err)
        got[v] = m
        ci = f"[{m['ci_lo']:.4f},{m['ci_hi']:.4f}]" if np.isfinite(m['ci_lo']) else "-"
        print(f"    {v:<19}{m['n']:>5}{m['n_eff']:>7.1f}{m['bias']:>10.4f}{m['rmse']:>10.4f}"
              f"{ci:>20}{m['med']:>9.4f}{m['p95']:>9.4f}{m['p99']:>9.4f}")

    # ---------- dynamic phase: describe the SIGNAL, never call it error ----------
    if ph_dyn.any():
        print(f"    -- DYNAMIC phase ({int(ph_dyn.sum())} exchanges): measured motion, "
              f"no ground truth --")
        print(f"    {'estimator':<19}{'n':>5}{'rms':>10}{'p95|v|':>10}{'peak|v|':>10}"
              f"{'span [m/s]':>22}")
        for v in VARIANTS:
            x = get_variant(d, v)
            if x is None:
                continue
            z = x[ph_dyn]
            z = z[np.isfinite(z)]
            if z.size == 0:
                continue
            span = f"{np.min(z):+.2f} .. {np.max(z):+.2f}"
            print(f"    {v:<19}{z.size:>5}{np.sqrt(np.mean(z**2)):>10.4f}"
                  f"{np.percentile(np.abs(z), 95):>10.4f}{np.max(np.abs(z)):>10.4f}"
                  f"{span:>22}")
        # The one honest accuracy number available while moving: the two radios
        # estimate the SAME quantity from different bursts, so their disagreement
        # bounds the error without needing to know the true velocity.
        if 'v_nodeA' in d and 'v_nodeB' in d:
            dz = (np.atleast_1d(d['v_nodeA']).astype(float)
                  - np.atleast_1d(d['v_nodeB']).astype(float))[ph_dyn]
            dz = dz[np.isfinite(dz)]
            if dz.size > 5:
                print(f"    node A vs node B disagreement while moving: "
                      f"rms {np.sqrt(np.mean(dz**2)):.4f}  p95 {np.percentile(np.abs(dz), 95):.4f} m/s"
                      f"   (truth-free error proxy, n={dz.size})")

    # ---------- per-node agreement: a check that needs NO ground truth ----------
    # v_nodeA and v_nodeB are the SAME physical quantity estimated by the two
    # different radios, each anchored at its own measurement epoch.  They must
    # agree.  How far they disagree measures the loss of reciprocity between the
    # DL and UL epochs -- which is exactly what multipath and drift destroy.
    dis = get_variant(d, 'v_node_disagree')
    if dis is not None:
        x = dis[keep]; x = x[np.isfinite(x)]
        if x.size:
            print(f"    node agreement: mean(v_A - v_B) = {x.mean():+.4f} m/s | "
                  f"rms {np.sqrt((x**2).mean()):.4f} | p95 {np.percentile(np.abs(x),95):.4f} m/s")

    # CFO cancellation ratio: naive one-way error vs the best corrected error
    base = 'v_oneway_nocorr'
    others = [(got[v]['rmse'], v) for v in got
              if not v.startswith('v_oneway') and not v.startswith('v_node')
              and np.isfinite(got[v]['rmse'])]
    if base in got and others and np.isfinite(got[base]['rmse']):
        best = min(others)
        ccr  = 20*np.log10(got[base]['rmse']/max(best[0], 1e-12))
        print(f"    CFO cancellation ratio ({base} -> {best[1]}): {ccr:.1f} dB")
        # v_oneway_nocorr applies NO frequency correction, so it reports the whole
        # LO offset as velocity and flatters this ratio.  A real one-way receiver
        # would AFC-lock, and a CONTINUOUSLY tracking one would null real Doppler
        # too -- one-way is unidentifiable, not merely inaccurate.  Say so, so the
        # number is never quoted without its caveat.
        print("      (baseline applies no frequency correction; see v_oneway_afc.)")
        print("       A tracking one-way AFC would also null real Doppler --")
        print("       one-way is unidentifiable, not merely inaccurate.")

    # ---------- LO offset, drift, SFO ----------
    eps = np.atleast_1d(d['eps_bv']).astype(float)
    tdl = np.atleast_1d(d['t_dl']).astype(float)
    tdl = tdl - tdl[0]
    if eps[keep].size > 2:
        drift = np.polyfit(tdl[keep], eps[keep], 1)[0]
        print(f"    eps_BV: mean {eps[keep].mean():+.1f} Hz "
              f"({eps[keep].mean()/fc*1e6:+.3f} ppm)"
              f" | span {eps[keep].max()-eps[keep].min():.1f} Hz | drift {drift:+.3f} Hz/s")
        print(f"    predicted SFO (one TCXO drives LO and sample clock): "
              f"{eps[keep].mean()/fc*fs:+.1f} Hz at {fs/1e6:.2f} MS/s")
        # Contribution 2: the half-sum bias is -D*g/2 (Eq. drift); compare that
        # prediction with what the drift-compensated estimator actually removed.
        _dc, _pl = get_variant(d, 'v_bidir_driftcomp'), get_variant(d, 'v_bidir_plain')
        if _dc is not None:
            _g = np.atleast_1d(d['gap_s']).astype(float)[keep]
            print(f"    drift bias: predicted -D*g/2 = {-drift*np.median(_g)/2:+.3f} Hz | "
                  f"bias removed by driftcomp {-np.nanmean((_dc-_pl)[keep])*hz_mps:+.3f} Hz")

    # ---------- reliability ----------
    attempted = int(exch.max() - exch.min() + 1)
    outage    = 1.0 - R/attempted if attempted > 0 else 0.0
    slips     = (np.atleast_1d(d['nslip_dl']).astype(float) +
                 np.atleast_1d(d['nslip_ul']).astype(float))
    navg      = float(d.get('NAVG', 1))
    gaps      = np.atleast_1d(d['gap_s']).astype(float)*1e3
    print(f"    outage {100*outage:.1f}% ({attempted-R}/{attempted} exchanges unusable)"
          f" | cycle-slip {100*slips.mean()/(2*navg*nsym):.2f}% of pilot symbols")
    print(f"    DL-UL gap: mean {gaps.mean():.1f} ms  "
          f"(min {gaps.min():.1f}, max {gaps.max():.1f})")
    # Link quality.  se_hz is the standard error of the CFO fit itself, i.e. the
    # uncertainty on the number we actually use -- a more direct quality measure than
    # SNR, which is only a proxy that still has to be translated through aperture and
    # pilot count.  Both were already logged per burst but never reported.
    def _q(k):
        v = np.atleast_1d(d[k]).astype(float)[keep]
        return v[np.isfinite(v)]
    if 'snr_dl' in d and 'snr_ul' in d:
        a_, b_ = _q('snr_dl'), _q('snr_ul')
        if a_.size and b_.size:
            print(f"    SNR [dB]      : DL mean {a_.mean():5.1f} min {a_.min():5.1f} "
                  f"p5 {np.percentile(a_,5):5.1f}  |  "
                  f"UL mean {b_.mean():5.1f} min {b_.min():5.1f} p5 {np.percentile(b_,5):5.1f}")
    if 'se_dl' in d and 'se_ul' in d:
        a_, b_ = _q('se_dl'), _q('se_ul')
        if a_.size and b_.size:
            hzm = hz_mps
            print(f"    CFO fit se[Hz]: DL med {np.median(a_):5.2f} p95 {np.percentile(a_,95):5.2f}"
                  f"  |  UL med {np.median(b_):5.2f} p95 {np.percentile(b_,95):5.2f}"
                  f"   (= {np.median(a_)/hzm:.4f} / {np.median(b_)/hzm:.4f} m/s)")
    # Where does the error actually come from?  The half-sum splits cleanly:
    #   fd_half = (coarse_DL+coarse_UL)/2  +  (resid_DL+resid_UL)/2
    # The residual is measured AFTER derotating by the coarse term, so a coarse error
    # should be ABSORBED by the residual (strong anti-correlation, total unaffected).
    # If that holds, the coarse estimator is NOT the floor, and the gap between the
    # measured spread and the one the fit standard errors predict is SE optimism --
    # correlated phase noise breaking the independent-residual assumption behind the SE.
    _ck = ('cfo_dl_coarse', 'cfo_ul_coarse', 'cfo_dl_resid', 'cfo_ul_resid')
    if all(k in d for k in _ck):
        cpart = (np.atleast_1d(d['cfo_dl_coarse']).astype(float)[keep] +
                 np.atleast_1d(d['cfo_ul_coarse']).astype(float)[keep]) / 2
        rpart = (np.atleast_1d(d['cfo_dl_resid']).astype(float)[keep] +
                 np.atleast_1d(d['cfo_ul_resid']).astype(float)[keep]) / 2
        tot = cpart + rpart
        if cpart.size > 3 and cpart.std() > 0 and rpart.std() > 0:
            cc = float(np.corrcoef(cpart, rpart)[0, 1])
            print("    CFO split [Hz]: coarse std {:7.3f} | residual std {:7.3f} | "
                  "total std {:7.3f}   corr(c,r) = {:+.3f}".format(
                      cpart.std(), rpart.std(), tot.std(), cc))
            if 'se_dl' in d and 'se_ul' in d:
                sa_ = np.atleast_1d(d['se_dl']).astype(float)[keep]
                sb_ = np.atleast_1d(d['se_ul']).astype(float)[keep]
                pred = float(np.sqrt(np.nanmedian(sa_)**2 + np.nanmedian(sb_)**2)/2)
                if pred > 0:
                    print("    fit-SE check  : total std {:.3f} Hz vs {:.3f} Hz "
                          "predicted by the fit SEs -> optimism factor {:.2f}x".format(
                              tot.std(), pred, tot.std()/pred))
    sa, sb = str(d.get('SERIAL_A', '?')), str(d.get('SERIAL_B', '?'))
    print(f"    boards: A={sa[:16]}... B={sb[:16]}...  gains TX={d.get('TX_GAIN','?')} "
          f"mode={str(d.get('GAIN_MODE','?'))}")
    if 'proc_ms' in d:
        pm = np.atleast_1d(d['proc_ms']).astype(float)
        rt = np.nanmean(pm)/max(gaps.mean(), 1e-9)
        print(f"    estimator cost: {np.nanmean(pm):.2f} ms per exchange (both directions)"
              f" | real-time factor {rt:.2f} (<1 = keeps up)")
    if 'temp_a' in d:
        ta = np.atleast_1d(d['temp_a']).astype(float); tb = np.atleast_1d(d['temp_b']).astype(float)
        if np.isfinite(ta).any():
            print(f"    board temp: A {np.nanmin(ta):.1f}->{np.nanmax(ta):.1f} C | "
                  f"B {np.nanmin(tb):.1f}->{np.nanmax(tb):.1f} C")
    print("    NOTE: exchanges within one run are correlated; use n_eff / the CI, not n.")

    if got:
        expD.append((float(gaps[keep].mean() if keep.any() else gaps.mean()), tag,
                     {v: got[v]['rmse'] for v in got}))

# ---------- Exp. D: RMSE vs DL-UL gap ----------
if len(expD) > 1:
    expD.sort()
    print("\n" + "="*78)
    print("EXP. D -- RMSE (m/s) vs DL-UL gap")
    print("="*78)
    print(f"{'gap [ms]':>10}  {'run':<16}" + "".join(f"{v:>19}" for v in VARIANTS))
    for g, tag, r in expD:
        row = "".join(f"{r[v]:>19.4f}" if v in r and np.isfinite(r[v]) else f"{'-':>19}"
                      for v in VARIANTS)
        print(f"{g:>10.1f}  {tag:<16}{row}")

# ---------- figures ----------
# SAMPLING NOTE: exchanges are NOT evenly spaced (the gap jitters and the periodic save
# stalls the loop), so everything is plotted against the MEASURED timestamps and every
# fit is least-squares, which handles uneven sampling natively.  Nothing is resampled.
# Every exchange is drawn, INCLUDING the calibration stretch, which is shaded so the
# before/after-calibration behaviour is readable rather than merely present.
if MAKE_PLOTS:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available -- skipped figures; pip install matplotlib)")
    else:
        for fn in FILES:
            d   = add_derived(loadmat(fn, squeeze_me=True))
            tag = str(d.get('RUN_TAG', os.path.basename(fn)))
            hz  = float(d['HZ_PER_MPS'])
            vt  = float(d['FD_INJECT_HW'])/hz
            fdt = float(d['FD_INJECT_HW'])
            nc  = int(d['N_CAL']) if 'N_CAL' in d else 0
            gv  = lambda k: np.atleast_1d(d[k]).astype(float)

            t  = (gv('t_dl_rel') if 't_dl_rel' in d else gv('t_dl') - gv('t_dl')[0])
            tu = (gv('t_ul_rel') if 't_ul_rel' in d else t + gv('gap_s'))
            cdl, cul = gv('cfo_dl'), gv('cfo_ul')
            eps = gv('eps_bv')
            t_cal = t[min(nc, t.size-1)] if nc > 0 else None   # calibration boundary

            R_ = t.size
            pstat, pdyn, pcal = phase_masks(d, R_, t, nc)
            mstat = pstat & ~pcal                 # held-out static = the scoring window
            if not mstat.any():
                mstat = np.zeros(R_, bool)
                mstat[min(nc, R_ - 1):] = True
            has_dyn = bool(pdyn.any())
            t_move  = float(t[pdyn][0]) if has_dyn else None
            # A phase-aware capture knows exactly how long it calibrated for; N_CAL is
            # only the fallback count for pre-phase captures, and using it here shaded
            # the ENTIRE run grey whenever the run was shorter than N_CAL exchanges.
            if 'CAL_S' in d:
                nc    = int(np.sum(pcal))
                _cs   = float(d['CAL_S'])
                t_cal = _cs if (_cs > 0 and nc > 0) else None

            fig, ax = plt.subplots(4, 2, figsize=(13, 14.5))
            if has_dyn:
                _sub = (f"{int(pstat.sum())} static + {int(pdyn.sum())} moving"
                        f"  --  true v = 0 in the STATIC phase only")
            else:
                _sub = f"true velocity {vt:+.2f} m/s"
            fig.suptitle(f"{tag}   ({len(t)} exchanges, {_sub})", fontsize=13, y=0.995)

            def mark_cal(a):
                """Shade the calibration stretch so before/after is unambiguous."""
                if t_cal is not None and nc > 0:
                    a.axvspan(t[0], t_cal, color='0.5', alpha=.16, lw=0)
                    a.axvline(t_cal, color='0.35', ls=':', lw=1.1)

            def mark_move(a):
                """Mark where the operator started moving.  Everything to the right of
                this line has NO ground truth and must never be scored as error."""
                if t_move is not None:
                    a.axvspan(t[0], t_move, color='C0', alpha=.07, lw=0)
                    a.axvline(t_move, color='C3', ls='--', lw=1.3)
                # operator key-press events (EVENT_T / EVENT_KEY, see acquisition/campaign.py)
                for te, ke in zip(np.atleast_1d(d.get('EVENT_T', [])),
                                  np.atleast_1d(d.get('EVENT_KEY', []))):
                    a.axvline(te, color='m', ls=':', lw=1.2)
                    a.text(te, a.get_ylim()[1], chr(int(ke)), color='m', fontsize=8, va='top')

            # (0,0) the two links folded onto each other -- the residual gap IS 2*f_d
            ax[0,0].plot(t,  cdl,  '.-', ms=3, lw=.7, color='C0', label='cfo_DL')
            ax[0,0].plot(tu, -cul, '.-', ms=3, lw=.7, color='C3', label='-cfo_UL')
            mark_cal(ax[0,0])
            ax[0,0].set_ylabel('CFO [Hz]'); ax[0,0].set_xlabel('t [s]')
            ax[0,0].set_title('cfo_DL vs -cfo_UL — the gap between them is 2·f_d',
                              fontsize=10)
            ax[0,0].legend(fontsize=8); ax[0,0].grid(alpha=.3)

            # (0,1) oscillator term and its drift
            ax[0,1].plot(t, eps, '.-', ms=3, lw=.7, color='C2')
            if t.size > 2:
                pf = np.polyfit(t, eps, 1)
                ax[0,1].plot(t, np.polyval(pf, t), 'k--', lw=1,
                             label=f'fit {pf[0]:+.3f} Hz/s')
                ax[0,1].legend(fontsize=8)
            mark_cal(ax[0,1])
            ax[0,1].set_ylabel(r'$\epsilon_{BV}$ [Hz]'); ax[0,1].set_xlabel('t [s]')
            ax[0,1].set_title('Oscillator offset and its drift', fontsize=10)
            ax[0,1].grid(alpha=.3)

            # (1,0) Doppler before / after calibration, with the velocity scale alongside.
            # The shaded span is where b_cal was still being estimated: inside it fd_cal
            # tracks fd_half exactly, and it steps away the moment calibration engages.
            for k, c in (('fd_half', 'C0'), ('fd_bwd', 'C5'),
                         ('fd_cal', 'C1'), ('fd_ta', 'C2')):
                if k in d:
                    ax[1,0].plot(t, gv(k), '.-', ms=3, lw=.7, color=c, label=k)
            ax[1,0].axhline(fdt, color='k', ls='--', lw=1, label=f'true {fdt:+.1f} Hz')
            mark_cal(ax[1,0]); mark_move(ax[1,0])
            ax[1,0].set_ylabel(r'$\hat{f}_d$ [Hz]'); ax[1,0].set_xlabel('t [s]')
            ax[1,0].set_title('Doppler before / after calibration'
                              + (f'  (grey = {nc} calibrating'
                                 + (', blue = static, red = motion starts)'
                                    if has_dyn else ')')
                                 if nc > 0 else ''), fontsize=10)
            ax[1,0].legend(fontsize=8, ncol=2); ax[1,0].grid(alpha=.3)
            sec = ax[1,0].secondary_yaxis('right',
                                          functions=(lambda h: h/hz, lambda v: v*hz))
            sec.set_ylabel('velocity [m/s]')

            # (1,1) THE ESTIMATED VELOCITY, side by side with the Doppler it comes
            # from.  The grey band is +/-1 sigma of THIS run's held-out static phase,
            # i.e. its own measured noise floor -- so any excursion outside the band
            # is motion the radios genuinely resolved, not scatter.
            vk = [(k, c) for k, c in (('v_driftcomp', 'C3'), ('v_ta', 'C2'),
                                      ('v_half', 'C0')) if k in d]
            for k, c in vk:
                ax[1,1].plot(t, gv(k), '.-', ms=3, lw=.7, color=c, label=k)
            if vk and mstat.sum() > 3:
                _b = gv(vk[0][0])[mstat]
                _b = _b[np.isfinite(_b)]
                if _b.size > 3:
                    _sd = float(np.std(_b))
                    ax[1,1].axhspan(-_sd, _sd, color='0.35', alpha=.22, lw=0,
                                    label=f'static floor +/-{_sd:.3f} m/s')
            ax[1,1].axhline(vt, color='k', ls='--', lw=1)
            mark_move(ax[1,1])
            ax[1,1].set_ylabel('estimated velocity [m/s]'); ax[1,1].set_xlabel('t [s]')
            ax[1,1].set_title('Estimated velocity  (band = this run static floor)',
                              fontsize=10)
            ax[1,1].legend(fontsize=8, ncol=2); ax[1,1].grid(alpha=.3)

            # (2,0) error distribution over the POST-calibration exchanges only --
            # including the calibrating stretch would flatter fd_half and penalise fd_cal
            for v, c in (('v_half', 'C0'), ('v_bwd', 'C5'),
                         ('v_cal', 'C1'), ('v_ta', 'C2')):
                if v in d:
                    e = np.abs(gv(v)[mstat] - vt)
                    e = np.sort(e[np.isfinite(e)])
                    if e.size:
                        ax[2,0].plot(e, np.linspace(0, 1, e.size), lw=1.4,
                                     color=c, label=v)
            ax[2,0].set_xscale('log'); ax[2,0].set_xlabel('|velocity error| [m/s]')
            ax[2,0].set_ylabel('CDF'); ax[2,0].legend(fontsize=8, loc='lower right')
            ax[2,0].set_title('Error CDF — STATIC phase only (where truth is known)'
                              if has_dyn else
                              'Error CDF (post-calibration) — tails, not just RMSE',
                              fontsize=10)
            ax[2,0].grid(alpha=.3)
            best = 'v_ta' if 'v_ta' in d else 'v_half'
            r = gv(best)[mstat] - vt; r = r[np.isfinite(r)]
            if r.size > 20:
                iax = ax[2,0].inset_axes([0.08, 0.55, 0.38, 0.40])
                iax.hist(r, bins=20, color='C2', alpha=.75)
                iax.axvline(0, color='k', lw=.8)
                iax.set_title(f'{best} residual', fontsize=7)
                iax.tick_params(labelsize=6)

            # (2,1) what limits the estimate right now?  Plot |error| against the
            # WORSE of the two directions' SNR -- the estimate uses both, so the weaker
            # one sets the quality.  A falling trend means SNR-limited (better link
            # budget buys accuracy); a flat trend means a floor has been reached and
            # more SNR buys nothing.
            drew = False
            if 'snr_dl' in d and 'snr_ul' in d and best in d:
                er_all = np.abs(gv(best)[mstat] - vt)
                for key, col, lab in (('snr_dl', 'C0', 'DL'), ('snr_ul', 'C1', 'UL')):
                    sn = gv(key)[mstat]
                    ok = np.isfinite(sn) & np.isfinite(er_all) & (er_all > 0)
                    sn_, er_ = sn[ok], er_all[ok]
                    if sn_.size <= 10:
                        continue
                    drew = True
                    ax[2,1].scatter(sn_, er_, s=7, alpha=.30, color=col,
                                    edgecolors='none', label=lab + ' SNR')
                    lo_, hi_ = np.percentile(sn_, [2, 98])
                    edges = np.linspace(lo_, hi_, 9)
                    bx, by = [], []
                    for i in range(len(edges)-1):
                        m_ = (sn_ >= edges[i]) & (sn_ < edges[i+1])
                        if m_.sum() >= 5:
                            bx.append(0.5*(edges[i]+edges[i+1]))
                            by.append(np.median(er_[m_]))
                    if len(bx) > 1:
                        ax[2,1].plot(bx, by, 'o-', color=col, lw=1.8, ms=4,
                                     markeredgecolor='k', markeredgewidth=.4,
                                     label=lab + ' binned median')
            if drew:
                ax[2,1].set_yscale('log')
                ax[2,1].set_xlabel('SNR [dB]')
                ax[2,1].set_ylabel('|velocity error| [m/s]')
                ax[2,1].set_title(best + ' error vs per-direction SNR — flat = floor',
                                  fontsize=10)
                ax[2,1].legend(fontsize=7, ncol=2); ax[2,1].grid(alpha=.3)
            else:
                ax[2,1].axis('off')

            # (3,0) where the error comes from.  The residual is measured AFTER the
            # coarse derotation, so a coarse error should be absorbed by the residual:
            # points on the -1 line mean the coarse estimator is NOT the floor.
            drew2 = False
            if all(k in d for k in ('cfo_dl_coarse', 'cfo_ul_coarse',
                                    'cfo_dl_resid', 'cfo_ul_resid')):
                cp = (gv('cfo_dl_coarse')[nc:] + gv('cfo_ul_coarse')[nc:]) / 2
                rp = (gv('cfo_dl_resid')[nc:] + gv('cfo_ul_resid')[nc:]) / 2
                ok = np.isfinite(cp) & np.isfinite(rp)
                cp, rp = cp[ok], rp[ok]
                if cp.size > 10 and cp.std() > 0:
                    drew2 = True
                    cc = float(np.corrcoef(cp, rp)[0, 1])
                    cp0, rp0 = cp - cp.mean(), rp - rp.mean()
                    ax[3,0].scatter(cp0, rp0, s=8, alpha=.4, color='C4',
                                    edgecolors='none')
                    lim = np.array([cp0.min(), cp0.max()])
                    ax[3,0].plot(lim, -lim, 'k--', lw=1,
                                 label='slope -1 = fully absorbed')
                    ax[3,0].set_xlabel('coarse part of f_d, mean removed [Hz]')
                    ax[3,0].set_ylabel('residual part, mean removed [Hz]')
                    ax[3,0].set_title('Coarse vs residual   corr = {:+.3f}'.format(cc),
                                      fontsize=10)
                    ax[3,0].legend(fontsize=8); ax[3,0].grid(alpha=.3)
            if not drew2:
                ax[3,0].axis('off')
            # (3,1) is the drift rate CONSTANT?  A rolling slope answers it -- a
            # time-varying rate is why one static b_cal goes stale and TA does not.
            W = 21
            if t.size > W + 2:
                mid, rate = [], []
                for i in range(t.size - W):
                    sl = slice(i, i + W)
                    rate.append(np.polyfit(t[sl], eps[sl], 1)[0])
                    mid.append(t[sl].mean())
                ax[3,1].plot(mid, rate, '-', lw=1, color='C4')
                ax[3,1].axhline(np.polyfit(t, eps, 1)[0], color='k', ls='--', lw=1,
                                label='whole-run fit')
                ax[3,1].legend(fontsize=8)
            mark_cal(ax[3,1])
            ax[3,1].set_ylabel('drift rate [Hz/s]'); ax[3,1].set_xlabel('t [s]')
            ax[3,1].set_title(f'Rolling drift rate ({W}-exchange window)', fontsize=10)
            ax[3,1].grid(alpha=.3)

            out = os.path.join(HERE, f"fig_{tag}.png")
            fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
            print(f"  figure -> {out}")
print()
