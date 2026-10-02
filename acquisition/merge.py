"""Combine the two laptops' captures into one file the analyser can read.

    python merge.py                          # auto-pairs the two newest here
    python merge.py capture_nodeA_X.mat capture_nodeB_X.mat

Node B received A->B (the DL); node A received B->A (the UL).  Pairing is by
exchange id, which both nodes computed as slot//2 from their own wall clock, so
no link was needed during the run.

t_dl comes from laptop B and t_ul from laptop A, so their difference carries the
offset between the two host clocks.  The two sit in consecutive slots, so the
excess over one SLOT_S IS that offset:  delta = median(t_ul - t_dl) - SLOT_S.
"""
import glob
import os
import sys
import numpy as np
from scipy.io import loadmat, savemat

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "captures"))   # all captures live here
os.makedirs(DATA, exist_ok=True)
args = sys.argv[1:]
if len(args) != 2:
    na = sorted(glob.glob(os.path.join(DATA, "capture_nodeA_*.mat")), key=os.path.getmtime)
    nb = sorted(glob.glob(os.path.join(DATA, "capture_nodeB_*.mat")), key=os.path.getmtime)
    if not na or not nb:
        sys.exit("need one capture_nodeA_*.mat and one capture_nodeB_*.mat")
    args = [na[-1], nb[-1]]
    print(f"auto-paired: {os.path.basename(args[0])} + {os.path.basename(args[1])}")

A, B = (loadmat(f, squeeze_me=True) for f in args)
if str(A.get('NODE', '')).strip() == 'B':                 # tolerate swapped arguments
    A, B = B, A

# Different numerology means the far burst was never demodulated the same way.
for k in ('FC', 'FS', 'FFTSIZE', 'DF', 'NSYM', 'RATIO', 'SLOT_S'):
    if abs(float(A[k]) - float(B[k])) > 1e-9:
        sys.exit(f"MISMATCH: {k} is {A[k]} on A but {B[k]} on B -- both nodes must "
                 "run identical config; re-run with the same config.py")

ea, eb = np.atleast_1d(A['exch_id']).astype(int), np.atleast_1d(B['exch_id']).astype(int)
common = np.intersect1d(ea, eb)
if common.size < 3:
    sys.exit(f"only {common.size} paired exchanges -- the host clocks were probably not "
             "aligned; run w32tm /resync on both and repeat")
# Explicit id->row maps: searchsorted would silently pair the wrong rows if either
# log were unsorted or held a duplicate id.
ia = np.array([{int(v): i for i, v in enumerate(ea)}[int(v)] for v in common])
ib = np.array([{int(v): i for i, v in enumerate(eb)}[int(v)] for v in common])

span = int(np.ptp(np.concatenate([ea, eb]))) + 1
eff = common.size / max(span, 1)
print(f"paired {common.size} exchanges (A {ea.size}, B {eb.size}) | "
      f"efficiency {100*eff:.1f}% of {span} slots")
if eff < 0.8:
    print("  *** under 80% paired: the laptop clocks were drifting apart. Resync both.")

g = lambda D, k, i: np.atleast_1d(D[k]).astype(float)[i]
SLOT = float(A['SLOT_S'])
t_dl, t_ul = g(B, 't_host', ib), g(A, 't_host', ia)
delta = float(np.median(t_ul - t_dl) - SLOT)
print(f"cross-laptop clock offset = {delta*1e3:+.1f} ms (slot {SLOT*1e3:.0f} ms)")
if abs(delta) > 0.4 * SLOT:
    print("  *** |delta| near half a slot: the slot assignment was breaking down.")
t_ul -= delta                                             # put both on B's clock

T0 = float(B['T0_EPOCH'])
rel = t_dl - T0

# ---- phases: static only where BOTH nodes said static ----------------------
# Each node timed its window from its own start and the laptops start at
# different moments, so the conservative intersection is the only safe rule --
# anything looser puts moving data into the calibration window.
pa, pb = g(A, 'phase', ia), g(B, 'phase', ib)
phase = np.maximum(pa, pb)
cal_a, cal_b = float(A.get('CAL_S', 0)), float(B.get('CAL_S', 0))
cal_m = (g(A, 't_rel', ia) < cal_a) & (g(B, 't_rel', ib) < cal_b) if cal_a and cal_b \
    else np.zeros(common.size, bool)
static_s = float(rel[phase == 0].max()) + 1e-6 if (phase == 0).any() else 0.0
cal_s = float(rel[cal_m].max()) + 1e-6 if cal_m.any() else 0.0
print(f"phases: {int((phase==0).sum())} static ({int(cal_m.sum())} calibrating) + "
      f"{int((phase==1).sum())} dynamic -> CAL_S={cal_s:.2f}s STATIC_S={static_s:.2f}s")
if static_s == 0.0 and float(A.get('STATIC_S', 0)) > 0:
    print("  note: the two static windows never overlapped on a paired exchange -- "
          "start both laptops before the static phase and repeat.")

out = dict(exch_id=common.astype(float), t_dl=t_dl, t_ul=t_ul, T0_EPOCH=T0,
           t_dl_rel=rel, t_ul_rel=t_ul - T0, gap_s=t_ul - t_dl, phase=phase,
           cfo_dl=g(B, 'cfo_total', ib), cfo_ul=g(A, 'cfo_total', ia),
           cfo_dl_coarse=g(B, 'cfo_coarse', ib), cfo_ul_coarse=g(A, 'cfo_coarse', ia),
           cfo_dl_resid=g(B, 'cfo_resid', ib), cfo_ul_resid=g(A, 'cfo_resid', ia),
           snr_dl=g(B, 'snr', ib), snr_ul=g(A, 'snr', ia),
           se_dl=g(B, 'se_hz', ib), se_ul=g(A, 'se_hz', ia),
           nslip_dl=g(B, 'nslip', ib), nslip_ul=g(A, 'nslip', ia),
           rssi_dl=g(B, 'rssi', ib), rssi_ul=g(A, 'rssi', ia),
           d0_dl=g(B, 'd0', ib), d0_ul=g(A, 'd0', ia),
           STATIC_S=static_s, CAL_S=cal_s, CLOCK_DELTA_S=delta,
           GAP_S=float(np.median(t_ul - t_dl)), N_CAL=float(cal_m.sum()),
           LO_OFFSET_B_HZ=float(B.get('LO_OFFSET_HZ', 0)) - float(A.get('LO_OFFSET_HZ', 0)),
           SERIAL_A=str(A.get('SERIAL', '?')), SERIAL_B=str(B.get('SERIAL', '?')),
           RUN_TAG=str(A.get('RUN_TAG', 'twolaptop')), NOTE=str(A.get('NOTE', '')))
for k in ('FC', 'FS', 'FFTSIZE', 'DF', 'NSYM', 'NPIL', 'RATIO', 'TSYM', 'HZ_PER_MPS',
          'FD_INJECT_HW', 'TX_GAIN', 'RX_GAIN', 'GAIN_MODE', 'pilot_active_idx'):
    if k in A:
        out[k] = A[k]
for src, key, idx in ((B, 'H_DL', ib), (A, 'H_UL', ia)):
    if 'H' in src:
        out[key] = np.atleast_2d(src['H'])[idx, :]

dst = os.path.join(DATA, f"capture_icc_{out['RUN_TAG'] or 'twolaptop'}.mat")
savemat(dst, out, do_compression=True)
print(f"\nwrote {dst}\n  now run:  python ../analyze_icc.py \"{dst}\"")
