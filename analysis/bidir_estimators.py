"""
bidir_estimators.py -- ONE definition of every velocity estimator, used by both the
campaign script (live print) and the analysis scripts (offline).

WHY THIS FILE EXISTS
--------------------
The estimator math used to be copy-pasted between icc_campaign_pluto.py,
analyze_icc.py, analyze_icc.m and TWO_LAPTOP/node_pluto.py.  Copies drift apart
silently (the frame-detection threshold already differs 0.001 vs 0.01 between two
of them).  Everything derived now lives here, is computed from the RAW logged
quantities, and therefore also applies to captures that were recorded before these
estimators existed.

THE MEASUREMENT
---------------
One exchange is two bursts, in this order on the wire:

    DL[k]           A transmits, B receives   -> B measures cfo_A2B[k] at t_A2B[k]
    UL[k]           B transmits, A receives   -> A measures cfo_B2A[k] at t_B2A[k]

Each node measures ONE number.  CFO is a pairwise quantity:

    cfo_A2B = f_d + f_lo          (measured at B)
    cfo_B2A = f_d - f_lo          (measured at A)

f_d is the Doppler (SAME sign both ways -- physical motion does not care which
radio is transmitting).  f_lo is the LO offset between the pair (OPPOSITE sign,
because swapping TX/RX role swaps which oscillator is subtracted).  That sign
asymmetry is the entire method:

    f_d   = (cfo_A2B + cfo_B2A)/2        <- half-SUM  kills the oscillator
    f_lo  = (cfo_A2B - cfo_B2A)/2        <- half-DIFF keeps only the oscillator

NEITHER NODE CAN DO THIS ALONE.  Node A only ever holds cfo_B2A; node B only ever
holds cfo_A2B.  One number must cross the link before any velocity exists.  That is
not an implementation shortcut, it is the structure of the problem -- and it is why
Jensen & Bokulic telemeter their counter values to the ground rather than solving
it onboard.

THE DRIFT TERM (why a static bias calibration cannot work)
----------------------------------------------------------
The two bursts are not simultaneous; UL happens a gap g after DL.  Carrying that
through:

    fd_plain = f_d + [f_lo(t_A2B) - f_lo(t_B2A)]/2 = f_d - D*g/2

where D = d f_lo/dt.  So the half-sum's bias is NOT an unknown hardware constant --
it is the drift rate times half the gap, and BOTH are measured every exchange.
That is what `driftcomp` below removes, with no calibration run of any kind.  It is
also why a constant b_cal goes stale: b_cal ~ D*g/2, and D decayed by 10x over a
single 14-minute run in the lab data.
"""

import numpy as np

# ---------------------------------------------------------------------------
# GLOSSARY.  Old capture-file names on the left, what they actually mean on the
# right.  Old names are still read and still written, so nothing already recorded
# is orphaned; new names are written alongside them.
# ---------------------------------------------------------------------------
RENAMES = {
    # raw per-direction measurements ---------------------------------------
    'cfo_dl':        ('cfo_A2B',      'CFO of the A->B burst, measured AT NODE B [Hz]'),
    'cfo_ul':        ('cfo_B2A',      'CFO of the B->A burst, measured AT NODE A [Hz]'),
    'cfo_dl_coarse': ('cfo_A2B_coarse','coarse (preamble) part of cfo_A2B [Hz]'),
    'cfo_ul_coarse': ('cfo_B2A_coarse','coarse (preamble) part of cfo_B2A [Hz]'),
    'cfo_dl_resid':  ('cfo_A2B_resid','residual (pilot-slope) part of cfo_A2B [Hz]'),
    'cfo_ul_resid':  ('cfo_B2A_resid','residual (pilot-slope) part of cfo_B2A [Hz]'),
    't_dl':          ('t_A2B',        'epoch node B measured the A->B burst [s]'),
    't_ul':          ('t_B2A',        'epoch node A measured the B->A burst [s]'),
    't_dl_rel':      ('t_A2B_rel',    't_A2B relative to run start [s]'),
    't_ul_rel':      ('t_B2A_rel',    't_B2A relative to run start [s]'),
    'gap_s':         ('gap_s',        't_B2A - t_A2B, the DL->UL measurement gap [s]'),
    'snr_dl':        ('snr_at_B',     'pilot-EVM SNR of the burst B received [dB]'),
    'snr_ul':        ('snr_at_A',     'pilot-EVM SNR of the burst A received [dB]'),
    'se_dl':         ('se_at_B',      'std error of B''s CFO slope fit [Hz]'),
    'se_ul':         ('se_at_A',      'std error of A''s CFO slope fit [Hz]'),
    # fused quantities ------------------------------------------------------
    'eps_bv':        ('f_lo_offset',  'LO offset between the pair, (A2B-B2A)/2 [Hz]'),
    'fd_half':       ('fd_plain',     'half-sum Doppler, no correction [Hz]'),
    'fd_cal':        ('fd_biascal',   'half-sum minus a STATIC bias constant [Hz]'),
    'fd_ta':         ('fd_tinterp',   'DL interpolated to the UL epoch, then half-sum [Hz]'),
    'v_1way':        ('v_oneway_nocorr','ONE direction''s TOTAL CFO read as if it were Doppler [m/s]'),
    'v_half':        ('v_bidir_plain', 'half-sum, no correction [m/s]'),
    'v_cal':         ('v_bidir_biascal','half-sum minus a static bias constant [m/s]'),
    'v_ta':          ('v_bidir_tinterp','time-interpolated half-sum [m/s]'),
    'b_cal':         ('bias_plain_hz', 'the static bias constant subtracted by fd_biascal [Hz]'),
    'b_cal_ta':      ('bias_tinterp_hz','the static bias constant subtracted by fd_tinterp [Hz]'),
}

# Estimators this module DERIVES (they exist in no capture file; they are computed
# from the raw per-direction numbers, so old captures get them too).
DERIVED = {
    'v_bidir_driftcomp': 'half-sum + D*g/2, D from the local f_lo slope. NO CALIBRATION.',
    'v_bidir_backpair':  'UL[k] paired with the FOLLOWING DL[k+1] instead of the preceding one.',
    'v_bidir_sympair':   'mean of the forward and backward pairings; drift artefacts cancel.',
    'v_nodeA':           "node A's own estimate, anchored at A's measurement epoch t_B2A.",
    'v_nodeB':           "node B's own estimate, anchored at B's measurement epoch t_A2B.",
    'v_node_disagree':   'v_nodeA - v_nodeB. Needs NO ground truth; flags loss of reciprocity.',
    'v_oneway_afc':      'one-way CFO after a one-time AFC lock at t=0 (fair one-way baseline).',
}

DRIFT_WINDOW = 3         # exchanges in the TRAILING f_lo slope fit.  Chosen 2026-09-28 on the
                         # GAP_0922 sweeps: the lag of a trailing fit grows with the window and
                         # dominates its noise at every gap (RMSE averaged over the 12 gap runs:
                         # W=3 0.042, 5 0.046, 9 0.052, 21 0.065 m/s); the static floor is the
                         # same for all W (0.030 m/s).


def _col(d, *names):
    """First present key among `names`, as a 1-D float array; None if none present."""
    for n in names:
        if n in d:
            return np.atleast_1d(d[n]).astype(float)
    return None


def local_drift(t, f_lo, window=DRIFT_WINDOW):
    """d f_lo/dt [Hz/s] per exchange, from a TRAILING (causal) least-squares slope.

    Exchange i uses exchanges i-window+1 .. i only, i.e. what the link has already
    measured when exchange i completes.  An earlier version used a CENTRED window,
    which reads window//2 exchanges from the future: it looked better at long gaps
    (no lag) but cannot run live.  The trailing fit estimates the slope about half a
    window in the past, so under a curving drift it lags by eps_ddot*(window-1)*T_u/2.

    The first window-1 exchanges are left NaN rather than fitted on a truncated
    window: the start of a run is where the warm-up curvature is worst."""
    t = np.asarray(t, float)
    f_lo = np.asarray(f_lo, float)
    D = np.full(t.size, np.nan)
    for i in range(window - 1, t.size):
        sl = slice(i - window + 1, i + 1)
        tt, ff = t[sl], f_lo[sl]
        ok = np.isfinite(tt) & np.isfinite(ff)
        if ok.sum() > 2:
            D[i] = np.polyfit(tt[ok], ff[ok], 1)[0]
    return D


MAX_SPAN_MULT = 3.0      # refuse to interpolate across a hole wider than this
                         # many times the median exchange spacing


def _interp_at(ts, vs, tq, max_span=None):
    """Linear interpolation of (ts, vs) at query epochs tq. NaN outside the span.

    Written explicitly rather than with np.interp for two reasons.  np.interp
    CLAMPS outside the range instead of returning NaN, which would fabricate an
    estimate for the first and last exchange out of a constant.  And np.interp will
    happily interpolate across an arbitrarily wide HOLE, which is worse.

    THE HOLE GUARD IS NOT COSMETIC.  The campaign loop stalls for 2-15 s every
    SAVE_EVERY exchanges while the capture is rewritten, and the LO drifts hundreds
    of Hz during that stall.  Interpolating a CFO across the hole invents a value
    that never existed, and on the lab data that single artefact was costing the
    time-interpolated estimators a factor of FOUR in RMSE (v_ta 0.2432 -> 0.0600
    m/s once stall-adjacent exchanges are excluded).  A straight line drawn through
    a 15-second gap in a drifting oscillator is not an estimate; it is a guess, and
    it should be NaN."""
    ts = np.asarray(ts, float); vs = np.asarray(vs, float); tq = np.asarray(tq, float)
    out = np.full(tq.size, np.nan)
    order = np.argsort(ts)
    ts_s, vs_s = ts[order], vs[order]
    if max_span is None and ts_s.size > 2:
        dt = np.diff(ts_s)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        max_span = MAX_SPAN_MULT * float(np.median(dt)) if dt.size else np.inf
    if max_span is None:
        max_span = np.inf
    for i, t in enumerate(tq):
        if not np.isfinite(t):
            continue
        j = np.searchsorted(ts_s, t)
        if 0 < j < ts_s.size:
            den = ts_s[j] - ts_s[j - 1]
            if abs(den) > 1e-12 and den <= max_span:
                a = (t - ts_s[j - 1]) / den
                out[i] = vs_s[j - 1] + a * (vs_s[j] - vs_s[j - 1])
    return out


def add_estimators(d, drift_window=DRIFT_WINDOW):
    """Add every derived estimator (and every new-style alias) to a loaded capture.

    Accepts a dict from scipy.io.loadmat(..., squeeze_me=True) using either the old
    or the new field names, and mutates it in place.  Safe to call twice."""
    hz = float(d['HZ_PER_MPS']) if 'HZ_PER_MPS' in d else None
    if hz is None:
        return d

    a2b = _col(d, 'cfo_A2B', 'cfo_dl')
    b2a = _col(d, 'cfo_B2A', 'cfo_ul')
    if a2b is None or b2a is None:
        return d
    t_a2b = _col(d, 't_A2B_rel', 't_dl_rel', 't_A2B', 't_dl')
    t_b2a = _col(d, 't_B2A_rel', 't_ul_rel', 't_B2A', 't_ul')
    gap = _col(d, 'gap_s')
    if gap is None and t_a2b is not None and t_b2a is not None:
        gap = t_b2a - t_a2b

    # --- new-style aliases for the raw columns (old names stay put) ---------
    for old, (new, _desc) in RENAMES.items():
        if old in d and new not in d:
            d[new] = d[old]

    f_lo = _col(d, 'f_lo_offset', 'eps_bv')
    if f_lo is None:
        f_lo = (a2b - b2a) / 2.0
        d['f_lo_offset'] = f_lo
    fd_plain = _col(d, 'fd_plain', 'fd_half')
    if fd_plain is None:
        fd_plain = (a2b + b2a) / 2.0
        d['fd_plain'] = fd_plain
    # The BASELINE every other method is compared against.  This used to appear
    # only if the capture file happened to carry a pre-computed v_half column, so
    # a capture that logs raw measurements alone silently lost its baseline row
    # from every results table.  Derive it here, where fd_plain already exists.
    d.setdefault('v_bidir_plain', fd_plain / hz)

    # Static-bias calibration is an ANALYSIS choice, not an acquisition one: which
    # exchanges count as calibration follows from the phase metadata.  Deriving it
    # here means the acquisition script never has to pick a bias, and an old
    # capture that already carries b_cal keeps its own value untouched.
    if 'fd_biascal' not in d and _col(d, 'fd_cal') is None:
        _b = _col(d, 'b_cal', 'bias_plain_hz')
        if _b is not None:
            _bias = float(np.atleast_1d(_b)[0])
        else:
            _m = np.zeros(fd_plain.size, dtype=bool)
            _cal_s = float(np.atleast_1d(d['CAL_S'])[0]) if 'CAL_S' in d else 0.0
            if _cal_s > 0 and t_a2b is not None:
                _m = np.asarray(t_a2b, dtype=float) < _cal_s
            elif 'N_CAL' in d:
                _m[:int(np.atleast_1d(d['N_CAL'])[0])] = True
            _f = fd_plain[_m & np.isfinite(fd_plain)]
            _bias = float(np.median(_f)) if _f.size else 0.0
        d['bias_plain_hz'] = _bias
        d['fd_biascal'] = fd_plain - _bias
        d['v_bidir_biascal'] = d['fd_biascal'] / hz

    # --- (1) DRIFT-COMPENSATED: fd_plain = f_d - D*g/2, so add D*g/2 back ---
    if t_a2b is not None and gap is not None:
        D = local_drift(t_a2b, f_lo, drift_window)
        d['drift_hz_per_s'] = D
        d['fd_driftcomp'] = fd_plain + D * gap / 2.0
        d['v_bidir_driftcomp'] = d['fd_driftcomp'] / hz

    # --- (2) BACKWARD and SYMMETRIC pairings -------------------------------
    fb = np.full(a2b.size, np.nan)
    if a2b.size > 1:
        fb[:-1] = (b2a[:-1] + a2b[1:]) / 2.0
    d['fd_backpair'] = fb
    d['v_bidir_backpair'] = fb / hz
    d['fd_sympair'] = (fd_plain + fb) / 2.0
    d['v_bidir_sympair'] = d['fd_sympair'] / hz
    # keep the names the older analysis scripts already look for
    d.setdefault('fd_bwd', fb); d.setdefault('v_bwd', fb / hz)
    d.setdefault('fd_sym', d['fd_sympair']); d.setdefault('v_sym', d['v_bidir_sympair'])

    # --- (3) PER-NODE estimates --------------------------------------------
    # Each node anchors at the epoch of ITS OWN measurement and interpolates the
    # peer's fed-back series to that epoch.  These are genuinely DIFFERENT
    # estimators: they carry different drift residuals, so comparing them is a
    # consistency check that needs no ground truth.
    if t_a2b is not None and t_b2a is not None:
        # One shared span limit from the A->B cadence, so node A and node B reject
        # exactly the same stalled exchanges and stay directly comparable.
        _dt = np.diff(np.sort(t_a2b[np.isfinite(t_a2b)]))
        _dt = _dt[_dt > 0]
        _span = MAX_SPAN_MULT*float(np.median(_dt)) if _dt.size else None
        d['interp_max_span_s'] = _span if _span else np.nan
        b2a_at_a2b = _interp_at(t_b2a, b2a, t_a2b, _span)   # peer series -> B's epoch
        a2b_at_b2a = _interp_at(t_a2b, a2b, t_b2a, _span)   # peer series -> A's epoch
        fd_B = (a2b + b2a_at_a2b) / 2.0
        fd_A = (a2b_at_b2a + b2a) / 2.0

        # --- STALL MASK -----------------------------------------------------
        # A wide bracket is only half the problem.  The campaign pauses 2-15 s
        # every SAVE_EVERY exchanges to rewrite the capture, and during that pause
        # the radios stop transmitting.  The FIRST exchange after the pause comes
        # back with the LO tens of Hz away from where the neighbouring exchanges
        # sit (measured: a 44 Hz step across one 0.49 s interval, against 2.3 Hz of
        # ordinary drift).  Its own half-sum is fine -- both directions moved
        # together -- but any estimator that INTERPOLATES between it and its
        # neighbour amplifies that step into a ~1.5 m/s spike.
        # So an exchange is tainted if the interval on EITHER side of it is
        # anomalous, not just if the interpolation bracket is wide.
        if _span:
            n_ = t_a2b.size
            dt_prev = np.zeros(n_); dt_next = np.zeros(n_)
            dt_prev[1:] = np.diff(t_a2b)
            dt_next[:-1] = np.diff(t_a2b)
            tainted = (dt_prev > _span) | (dt_next > _span)
            d['stall_adjacent'] = tainted.astype(float)
            # The two nodes interpolate in OPPOSITE temporal directions, so a given
            # tainted sample lands on different exchanges for each:
            #   node A at k spans {k, k+1}   -> invalid if k or k+1 is tainted
            #   node B at k spans {k-1, k}   -> invalid if k-1 or k is tainted
            # Using one symmetric mask for both (the obvious thing) leaves node B's
            # spikes completely untouched -- they sit one exchange later than A's.
            badA = tainted.copy(); badA[:-1] |= tainted[1:]
            badB = tainted.copy(); badB[1:]  |= tainted[:-1]
            fd_A = np.where(badA, np.nan, fd_A)
            fd_B = np.where(badB, np.nan, fd_B)
        d['fd_nodeB'] = fd_B;  d['v_nodeB'] = fd_B / hz
        d['fd_nodeA'] = fd_A;  d['v_nodeA'] = fd_A / hz
        d['v_node_disagree'] = (fd_A - fd_B) / hz
        # node A's estimate IS the historical fd_ta/v_ta (A->B interpolated to the
        # B->A epoch).  Overwrite rather than setdefault: the value logged live by
        # the campaign predates the stall mask above, so deferring to it would keep
        # the artefact in the one column most damaged by it.
        d['fd_tinterp'] = fd_A
        d['v_bidir_tinterp'] = fd_A / hz

    # --- (4) a FAIR one-way baseline ---------------------------------------
    # v_oneway_nocorr applies NO frequency correction, so it reports the whole LO
    # offset as velocity.  A real one-way receiver would at least AFC-lock once.
    n0 = min(50, a2b.size)
    d['v_oneway_afc'] = (a2b - np.mean(a2b[:n0])) / hz
    d.setdefault('v_oneway_nocorr', a2b / hz)
    # legacy names too: captures log raw columns only, and the analyser's
    # plots still read eps_bv / fd_half / v_ta / ...
    for old, (new, _) in RENAMES.items():
        if new in d and old not in d:
            d[old] = d[new]
    if 'v_bidir_driftcomp' in d:
        d.setdefault('v_driftcomp', d['v_bidir_driftcomp'])
    return d


# Order the analysis prints them in: worst/naive first, best last.
VARIANT_ORDER = [
    'v_oneway_nocorr',      # no correction at all
    'v_oneway_afc',         # one-way with a one-time AFC lock
    'v_bidir_plain',        # half-sum
    'v_bidir_backpair',
    'v_bidir_sympair',
    'v_bidir_biascal',      # static bias constant
    'v_bidir_tinterp',      # time-interpolated  (== v_nodeA)
    'v_nodeA',
    'v_nodeB',
    'v_bidir_driftcomp',    # drift-compensated, NO calibration
]

LEGACY_ALIAS = {'v_oneway_nocorr': 'v_1way', 'v_bidir_plain': 'v_half',
                'v_bidir_biascal': 'v_cal',  'v_bidir_tinterp': 'v_ta',
                'v_bidir_backpair': 'v_bwd', 'v_bidir_sympair': 'v_sym'}


def get_variant(d, name):
    """Fetch an estimator by its new name, falling back to the legacy name."""
    v = _col(d, name, LEGACY_ALIAS.get(name, name))
    return v


def describe(name):
    """One-line explanation of any field name, old or new."""
    for old, (new, desc) in RENAMES.items():
        if name in (old, new):
            return desc
    return DERIVED.get(name, '(no description)')


if __name__ == '__main__':
    print(__doc__)
    print("\nFIELD NAMES  (old -> new)\n" + "=" * 78)
    for old, (new, desc) in RENAMES.items():
        print(f"  {old:<18} -> {new:<20} {desc}")
    print("\nDERIVED BY THIS MODULE\n" + "=" * 78)
    for k, v in DERIVED.items():
        print(f"  {k:<22} {v}")
