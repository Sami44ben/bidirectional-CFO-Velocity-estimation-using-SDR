"""All configuration, in one place, shared by every script.

campaign.py and node.py both import this, so the single-host rig and the
two-laptop rig cannot run different numerology -- which is exactly how they
drifted to 2.3 vs 3.5 GHz and NSYM 140 vs 64 in the previous version.

Any UPPERCASE name here is overridable from the environment as ICC_<NAME>,
so run_matrix-style sweeps never rewrite this file.
"""
import os

# ---- radios -----------------------------------------------------------------
SDR_A   = '104473f4a313000703002100aef2d4f228'   # serial or 'usb:1.2.3'
SDR_B   = '10447318ac0f001621001100998717342d'
OFFLINE = False          # True = numpy-only self-test, no hardware

# ---- RF ---------------------------------------------------------------------
FC        = 2.3e9        # carrier (Hz)
TX_GAIN   = -5           # dB
RX_GAIN   = 65           # dB, used only when GAIN_MODE == 'manual' fast_attack
GAIN_MODE = 'manual'

# ---- OFDM -------------------------------------------------------------------
FFTSIZE = 64
DF      = 60e3            # subcarrier spacing; coarse-CFO range is +-DF/2 per lag
NSYM    = 140            # symbols per burst = the Doppler observation aperture
NACT    = 60            # active subcarriers

RATIO   = 2              # every RATIO-th active subcarrier is a pilot
FS      = int(DF * FFTSIZE)
CP      = FFTSIZE // 8

# ---- run --------------------------------------------------------------------
RUN_TAG    = 'MOVE_REF_test2'      # overridden per run by ICC_RUN_TAG (run_matrix.py) or the env
MAX_ROUNDS = 0        # 0 = run until Ctrl+C
GAP_S      = 0.0         # extra dwell between the DL and UL measurement (s)
NOTE       = 'using the chaire_take one'

# ---- ground truth / injected quantities -------------------------------------
FD_INJECT = 0.0          # Hz pre-rotated on BOTH transmitters (motion-like: no sign flip)
LO_OFFSET = 0.0          # Hz detune on node B only (oscillator-like: flips sign)

# ---- run phases -------------------------------------------------------------
# Hold still for STATIC_S, then move.  The first CAL_FRAC of that window is used
# for calibration; the rest is held out and becomes the reported error floor.
# Calibrating and scoring on the same samples measures how well a median fits its
# own data, not how well the estimator works.
STATIC_S = 60.0
CAL_FRAC = 0.5

# ---- logging ----------------------------------------------------------------
SAVE       = True        # False = print only, write nothing (was False during the RC-car tests)
SAVE_EVERY = 25          # exchanges between background checkpoints (0 = only at end)
LOG_PILOTS = True        # per-symbol pilot matrix; ~270 kB/exchange
LOG_IQ_N   = 0           # raw IQ for the first N exchanges; ~2 MB each

# ---- acquisition ------------------------------------------------------------
RX_FLUSH = 4     # stale RX buffers dropped before the measured one.  Each is a
                 # full RXBUF USB transfer, so this is the biggest single lever
                 # on exchange rate -- and if it is too low the measured buffer
                 # can still straddle the moment TX started, which shows up as
                 # occasional huge pilot-fit standard errors, not as "no frame".

# ---- two-laptop only --------------------------------------------------------
NODE   = 'A'             # 'A' or 'B'; set per laptop
SLOT_S = 0.5             # TDD slot length; must match on both laptops

# ---- environment overrides --------------------------------------------------
def _env(name, val):
    s = os.environ.get('ICC_' + name)
    if s is None:
        return val
    if isinstance(val, bool):
        return bool(int(float(s)))
    if isinstance(val, int):
        return int(float(s))
    if isinstance(val, float):
        return float(s)
    return s


for _k, _v in list(globals().items()):
    if _k.isupper():
        globals()[_k] = _env(_k, _v)

# An injected Doppler is a known truth on EVERY exchange, so there is no static
# stretch to hold out.  Forced here rather than trusting a sweep to pass it.
if FD_INJECT != 0.0:
    STATIC_S = 0.0
CAL_S      = STATIC_S * CAL_FRAC
HZ_PER_MPS = FC / 3e8
TSYM       = (FFTSIZE + CP) / FS
