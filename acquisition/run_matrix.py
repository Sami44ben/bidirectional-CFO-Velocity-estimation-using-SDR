"""
run_matrix.py -- drive campaign.py through an automated sweep.

Nothing here edits the campaign script; overrides are passed as environment
variables, so a batch run can never silently write to the wrong RUN_TAG.

  python run_matrix.py warm            # hold the radios streaming until drift settles
  python run_matrix.py sep             # the Doppler x LO separation matrix
  python run_matrix.py gap             # the DL-UL gap sweep
  python run_matrix.py sep --dry       # print the full schedule, touch no hardware
  python run_matrix.py cal             # (optional) legacy static bias calibration

NO MANUAL STEP IS REQUIRED ANY MORE
-----------------------------------
The old version made you run `cal`, read b_cal / b_cal_ta off the terminal and
paste them in before `sep` would run.  That is gone, because the estimator the
analysis now leads with -- v_bidir_driftcomp -- needs no calibration at all:

    fd_plain = f_d - D*g/2      D = d(f_lo)/dt , g = the measured DL-UL gap

Both D and g are measured every exchange, so the half-sum's bias is COMPUTED, not
calibrated.  On the lab data that scores 0.0712 m/s against 0.2930 m/s for the
static-bias estimator it replaces.  `cal` is kept only so the older
v_bidir_biascal column stays reproducible; the sweep does not need it.

TWO BUGS THIS FILE USED TO HAVE
-------------------------------
1. CARRIER.  HZ_PER_MPS was hard-coded to 3.5e9/3e8 = 11.67 while the campaign
   ran at 2.3 GHz (7.67).  Every velocity in FD_LIST was therefore injected 52%
   too large and every run tag was mislabelled.  The carrier is now READ FROM THE
   CAMPAIGN SCRIPT, so the two cannot disagree again.

2. ORDERING.  The loop was `for lo in LO_LIST: for v in FD_LIST:`, which puts the
   LO axis in lockstep with warm-up drift.  Measured on capture_A_OTA_ST_CAL:
   eps_BV moves up to 3052 Hz inside ONE LO block and 4866 Hz across the whole
   sweep -- 2.4x the +/-2000 Hz LO step being injected.  Worse, f_D stepped
   monotonically inside each block while the drift fell monotonically, which
   manufactures a spurious cross-leakage of b_LO = -3.85 where the experiment
   wants 0.  Conditions are now RANDOMISED (fixed seed) and bracketed by repeated
   (0,0) reference points, so drift becomes noise instead of bias and what is
   left can be regressed out.
"""
import os
import random
import re
import subprocess
import sys
import time

HERE     = os.path.dirname(os.path.abspath(__file__))
POC      = HERE                                  # campaign.py lives next to this file
CAMPAIGN = os.path.join(POC, "campaign.py")      # the stall-free chain
CONFIG   = os.path.join(POC, "config.py")
PY       = sys.executable


# ---------------------------------------------------------------------------
# CARRIER -- read from the campaign script so there is exactly ONE definition.
# ---------------------------------------------------------------------------
def _campaign_fc():
    """Parse FC out of config.py.

    Deliberately fails loudly rather than falling back to a default: a silent
    default is exactly how this file ended up injecting 3.5 GHz velocities into a
    2.3 GHz campaign for an entire measurement session."""
    try:
        src = open(CONFIG, encoding="utf-8").read()
    except OSError as e:
        sys.exit(f"cannot read {CONFIG}: {e}")
    m = re.search(r"^FC\s*=\s*([0-9][0-9_.eE+-]*)", src, re.M)
    if not m:
        sys.exit("could not find 'FC = ...' in config.py -- refusing to "
                 "guess the carrier.  Fix the parse or set FC_OVERRIDE_HZ below.")
    return float(m.group(1))


FC_OVERRIDE_HZ = None                     # set only to deliberately differ from the campaign
C_LIGHT    = 3e8
FC_HZ      = FC_OVERRIDE_HZ if FC_OVERRIDE_HZ else _campaign_fc()
HZ_PER_MPS = FC_HZ / C_LIGHT              # 7.67 Hz per m/s at 2.3 GHz

# ---- how long each condition runs ------------------------------------------
EXCH_S      = 0.16       # seconds per exchange at 3.84 MS/s with RX_FLUSH=4 (measured 6.2/s);
                         # was 0.46 at 20.48 MS/s.  Every duration below scales with it.
N_WARM_MIN  = 15         # minutes of `warm` before a sweep (drift is worst when cold)
N_CAL_RUN   = 60         # exchanges for the optional legacy calibration run
N_PER_COND  = 300        # exchanges per condition in the sep matrix (~27 s each)
N_PER_GAP   = 120        # exchanges per gap point (the 4 s dwell run takes ~8 min)
REF_EVERY   = 4          # insert a (0,0) reference condition this often
SEED        = 20260904   # fixed, so the randomised order is reproducible and citable
ENV_COND    = "LOS"
ENV_DIST_M  = 1.0        # antenna separation, metres

# ---- optional legacy static bias (NOT needed; v_bidir_driftcomp ignores it) --
B_CAL    = None
B_CAL_TA = None

# 5 Dopplers x 3 LO offsets.  FD is the SAME sign both directions (motion-like);
# LO flips sign between DL and UL (oscillator-like).  Clean separation shows f_d
# tracking only FD, and f_lo tracking only LO.
FD_LIST  = [-10.0, -5.0, 0.0, +5.0, +10.0]      # m/s -> converted to Hz with HZ_PER_MPS
LO_LIST  = [0.0, +2000.0, -2000.0]              # Hz of LO detune on node B
GAP_LIST = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]      # s of EXTRA dwell between DL and UL

# Every sweep gets its own stamp, so re-running one never overwrites the last.
SESSION = time.strftime("%m%d_%H%M")


def run(tag, note, rounds, fd_hz=0.0, lo_hz=0.0, gap_s=0.0, n_cal=0,
        b_cal=None, b_cal_ta=None, dry=False, seq=None, nseq=None):
    env = dict(os.environ)
    env.update({
        "ICC_RUN_TAG":    tag,
        "ICC_NOTE":       note,
        "ICC_ENV_COND":   ENV_COND,
        "ICC_ENV_DIST":   str(ENV_DIST_M),
        "ICC_FD_INJECT":  str(fd_hz),
        "ICC_LO_OFFSET":  str(lo_hz),
        "ICC_GAP_S":      str(gap_s),
        "ICC_N_CAL":      str(n_cal),
        "ICC_MAX_ROUNDS": str(rounds),
        "ICC_B_CAL":      "none" if b_cal    is None else str(b_cal),
        "ICC_B_CAL_TA":   "none" if b_cal_ta is None else str(b_cal_ta),
        "ICC_SAVE_IQ":    "0",          # keep the sweep's files small
        "ICC_STATIC_S":   "0",          # sweep runs have no static/dynamic split
    })
    pos = f"[{seq}/{nseq}] " if seq else ""
    print(f"\n=== {pos}{tag}  fd={fd_hz:+.2f} Hz ({fd_hz/HZ_PER_MPS:+.2f} m/s)  "
          f"lo={lo_hz:+.0f} Hz  gap={gap_s:.2f}s  rounds={rounds}")
    if dry:
        return
    t0 = time.time()
    r = subprocess.run([PY, CAMPAIGN], env=env)
    if r.returncode != 0:
        print(f"  !! {tag} exited {r.returncode} -- stopping the sweep")
        sys.exit(r.returncode)
    print(f"  done in {time.time()-t0:.0f}s")


def _schedule(conditions, ref):
    """Randomise the condition order and bracket it with reference points.

    Randomising converts warm-up drift from a BIAS aligned with an experimental
    axis into NOISE spread across all of them.  The repeated (0,0) reference lets
    the analysis measure whatever drift is left and regress it out, because the
    reference conditions differ from each other ONLY by when they ran."""
    rng = random.Random(SEED)
    order = list(conditions)
    rng.shuffle(order)
    out = [ref]                                   # always open with a reference
    for i, c in enumerate(order):
        out.append(c)
        if (i + 1) % REF_EVERY == 0 and i + 1 < len(order):
            out.append(ref)

    out.append(ref)                               # ...and close with one
    return out


def _tag(kind, seq, fd_mps, lo_hz):
    """Tag carries the SEQUENCE INDEX as well as the condition.

    Without it the repeated (0,0) reference points -- and a (0,0) that is also a
    matrix condition in its own right -- would all resolve to one filename and
    silently overwrite each other.  With it, every run in the sweep is its own
    file, and the index is what lets the analysis regress the residual drift out
    (reference points differ from one another ONLY by when they ran)."""
    body = (f"v{fd_mps:+05.1f}_lo{lo_hz:+05.0f}"
            .replace('.', 'p').replace('+', 'p').replace('-', 'm'))
    return f"{kind}_{SESSION}_s{seq:02d}_{body}"


def main():
    what = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    dry  = "--dry" in sys.argv
    if "--offline" in sys.argv:           # simulate the whole sweep, no radios
        os.environ["ICC_OFFLINE"] = "1"
    if what not in ("cal", "sep", "gap", "warm"):
        sys.exit(__doc__)

    print(f"carrier {FC_HZ/1e9:.3f} GHz  ->  {HZ_PER_MPS:.3f} Hz per m/s "
          f"(read from config.py)")
    print(f"session stamp {SESSION}  |  order seed {SEED}")

    # ---------------- warm ----------------
    if what == "warm":
        # Drift decays from about -24 Hz/s to -2.5 Hz/s over the first quarter hour
        # and is still falling; starting a sweep cold puts the steepest, most
        # time-varying part of the transient inside the measurement.
        rounds = int(N_WARM_MIN * 60 / EXCH_S)    # EXCH_S per exchange, measured
        print(f"\nwarming the radios under realistic load for ~{N_WARM_MIN} min "
              f"({rounds} exchanges); this capture IS the static-floor run.")
        env = dict(os.environ)
        env.update({"ICC_RUN_TAG": f"WARM_{SESSION}", "ICC_STATIC_S": str(N_WARM_MIN*60),
                    "ICC_CAL_FRAC": str(1.0/N_WARM_MIN),
                    "ICC_MAX_ROUNDS": str(rounds), "ICC_N_CAL": "0",
                    "ICC_LOG_PILOTS": "1",          # the static floor doubles as the pilot-density source
                    "ICC_B_CAL": "none", "ICC_B_CAL_TA": "none", "ICC_SAVE_IQ": "0"})
        if dry:
            print("  (--dry: not started)")
            return
        subprocess.run([PY, CAMPAIGN], env=env)
        print("\nwarm-up done.  Now run:  python run_matrix.py sep")
        return

    # ---------------- cal (optional, legacy) ----------------
    if what == "cal":
        print("\nNOTE: this is only needed to reproduce the legacy v_bidir_biascal "
              "column.\n      v_bidir_driftcomp needs no calibration -- you can skip "
              "straight to `sep`.")
        run(f"CAL_{SESSION}_static", "static calibration; nothing injected",
            N_CAL_RUN, n_cal=30, dry=dry)
        print("\nIf you want the legacy column, read b_cal / b_cal_ta from the run "
              "above and paste them into B_CAL / B_CAL_TA in this file.")
        return

    # B_CAL is no longer required.  If it is unset the campaign refuses to
    # auto-calibrate whenever a Doppler is injected (it would absorb the ground
    # truth) and uses b_cal = 0, which leaves v_bidir_driftcomp untouched.
    if B_CAL is None or B_CAL_TA is None:
        print("\nB_CAL / B_CAL_TA are unset -- proceeding.  v_bidir_plain and\n"
              "v_bidir_driftcomp are unaffected; only the legacy v_bidir_biascal\n"
              "column will read as uncalibrated.")

    # ---------------- sep ----------------
    if what == "sep":
        conds = [(v, lo) for lo in LO_LIST for v in FD_LIST]
        sched = _schedule(conds, (0.0, 0.0))
        est_s = len(sched) * N_PER_COND * EXCH_S
        print(f"\nseparation matrix: {len(conds)} conditions "
              f"({len(FD_LIST)} Dopplers x {len(LO_LIST)} LO offsets)")
        print(f"  + {len(sched)-len(conds)} interleaved (0,0) reference points "
              f"= {len(sched)} runs of {N_PER_COND} exchanges")
        print(f"  randomised order, seed {SEED}; estimated {est_s/60:.0f} min\n")
        print("  schedule:")
        for i, (v, lo) in enumerate(sched, 1):
            mark = "  <- reference" if (v, lo) == (0.0, 0.0) else ""
            print(f"    {i:3d}.  v={v:+6.1f} m/s   LO={lo:+7.0f} Hz{mark}")
        for i, (v, lo) in enumerate(sched, 1):
            fd = v * HZ_PER_MPS
            run(_tag("SEP", i, v, lo),
                f"separation matrix seq {i}/{len(sched)}: v={v:+.1f} m/s, "
                f"LO={lo:+.0f} Hz, seed {SEED}",
                N_PER_COND, fd_hz=fd, lo_hz=lo,
                b_cal=B_CAL, b_cal_ta=B_CAL_TA, dry=dry, seq=i, nseq=len(sched))
        print(f"\nnow run:  python analyze_separation.py "
              f"\"../data/captures/*capture_icc_SEP_{SESSION}_*.mat\"")

    # ---------------- gap ----------------
    if what == "gap":
        # The gap sweep gets the same treatment: gap is confounded with time just
        # as badly as LO was, and this is now the headline experiment (the fan
        # data shows cancellation collapsing once the gap exceeds the channel
        # coherence time), so it must not carry a drift artefact.
        sched = _schedule(list(GAP_LIST), 0.0)
        est_s = sum(N_PER_GAP * (EXCH_S + g) for g in sched)
        print(f"\ngap sweep: {len(GAP_LIST)} points + "
              f"{len(sched)-len(GAP_LIST)} references = {len(sched)} runs "
              f"of {N_PER_GAP} exchanges")
        print(f"  randomised order, seed {SEED}; estimated {est_s/60:.0f} min")
        print("  TIP: run `python run_matrix.py warm` first unless you specifically")
        print("       want the cold-start transient in the data.\n")
        print("  schedule:")
        for i, g in enumerate(sched, 1):
            mark = "  <- reference" if g == 0.0 else ""
            print(f"    {i:3d}.  extra dwell {g:.2f} s{mark}")
        for i, g in enumerate(sched, 1):
            run(f"GAP_{SESSION}_s{i:02d}_{int(round(g*1000)):05d}ms",
                f"gap sweep seq {i}/{len(sched)}: extra dwell {g:.2f} s, seed {SEED}",
                N_PER_GAP, gap_s=g, b_cal=B_CAL, b_cal_ta=B_CAL_TA,
                dry=dry, seq=i, nseq=len(sched))
        print(f"\nnow run:  python ../analysis/analyze_icc.py \"../data/captures/*capture_icc_GAP_{SESSION}_*.mat\"")
        print("  (the Exp. D table appears at the end)")


if __name__ == "__main__":
    main()
