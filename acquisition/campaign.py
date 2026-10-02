"""Single-host bidirectional campaign: one laptop, both radios.

    python campaign.py            # uses config.py

Each exchange measures A->B then B->A and logs the two RAW per-direction CFOs.
It does NOT compute half-sums, calibrated biases or drift corrections: those are
estimators, they live in the analyser, and having one implementation of each is
why this file is short.
"""
import os
import sys
import time
import numpy as np
import config as C
import dsp  
import log as L
try:
    import msvcrt                      # Windows: non-blocking key check for event stamps
except ImportError:
    msvcrt = None

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "captures"))   # all captures live here
os.makedirs(DATA, exist_ok=True)
PATH = os.path.join(DATA, f"{'OFFLINE_' if C.OFFLINE else ''}capture_icc_{C.RUN_TAG}.mat")

# ---- radios -----------------------------------------------------------------
sdr_a = sdr_b = None
serial_a = serial_b = 'offline'
if not C.OFFLINE:
    import radio
    found = radio.scan()                       # scan ONCE, before claiming contexts
    print(f"found {len(found)} Pluto(s): {[s for _, s in found]}")
    sdr_a = radio.open(C.SDR_A, 0.0, found)
    sdr_b = radio.open(C.SDR_B, C.LO_OFFSET, found)   # B carries the LO detune
    serial_a, serial_b = C.SDR_A, C.SDR_B

L.header(C, PATH)

T0 = time.time()
meta = dict(FS=C.FS, FC=C.FC, FFTSIZE=C.FFTSIZE, DF=C.DF, NSYM=C.NSYM, RATIO=C.RATIO,
            NPIL=dsp.NPIL, TSYM=C.TSYM, HZ_PER_MPS=C.HZ_PER_MPS, GAP_S=C.GAP_S,
            FD_INJECT_HW=C.FD_INJECT, LO_OFFSET_B_HZ=C.LO_OFFSET, RUN_TAG=C.RUN_TAG,
            NOTE=C.NOTE, T0_EPOCH=T0, STATIC_S=C.STATIC_S, CAL_FRAC=C.CAL_FRAC,
            CAL_S=C.CAL_S, SAVE_EVERY=C.SAVE_EVERY, TX_GAIN=C.TX_GAIN,
            RX_GAIN=C.RX_GAIN, GAIN_MODE=C.GAIN_MODE, SERIAL_A=serial_a,
            SERIAL_B=serial_b, pilot_active_idx=dsp.PILOT_POS)
lg = L.Log(PATH, meta, heavy=('H_DL', 'H_UL', 'Rp_DL', 'Rp_UL', 'IQ_DL', 'IQ_UL'))

HDR = ("\n   idx      t[s]  ph     v[m/s]     f_lo[Hz]   SNR B/A [dB]\n  " + "-" * 58)
print(HDR)


def measure(sign, tx_sdr, rx_sdr):
    """One direction. Returns (estimate dict or None, midpoint timestamp)."""
    t0 = time.time()
    buf = dsp.fake_capture(sign) if C.OFFLINE else radio.burst(tx_sdr, rx_sdr, C.FD_INJECT)
    return dsp.estimate(buf), 0.5 * (t0 + time.time())


rnd = n_bad = 0
t_move = None
try:
    while not C.MAX_ROUNDS or rnd < C.MAX_ROUNDS:
        rnd += 1
        dl, t_dl = measure(+1, sdr_a, sdr_b)          # A -> B, measured at B
        if C.GAP_S:
            time.sleep(C.GAP_S)
        ul, t_ul = measure(-1, sdr_b, sdr_a)          # B -> A, measured at A
        if dl is None or ul is None:
            n_bad += 1
            print(f"[{rnd:5d}] no frame ({n_bad} so far)")
            continue

        # ---- run phase: 0 = static (truth is v=0), 1 = dynamic (moving) ----
        rel = t_dl - T0
        static = C.STATIC_S > 0 and rel < C.STATIC_S
        if C.STATIC_S > 0 and not static and t_move is None:
            t_move = rel
            print("\n" + "=" * 58)
            print(f"  STATIC PHASE DONE ({rel:.1f} s)   ***  MOVE NOW  ***")
            print("=" * 58 + "\n" + HDR)

        # ---- event stamp: any key press is logged as (t, key) with this exchange ----
        if msvcrt and msvcrt.kbhit():
            k = msvcrt.getwch()
            lg.meta.setdefault('EVENT_T', []).append(rel)
            lg.meta.setdefault('EVENT_KEY', []).append(ord(k))
            print(f"  *** event '{k}' stamped at t={rel:.2f}s ***")

        lg.add(exch_id=rnd, t_dl=t_dl, t_ul=t_ul, t_dl_rel=rel, t_ul_rel=t_ul - T0,
               gap_s=t_ul - t_dl, phase=0 if static else 1,
               cfo_dl=dl['cfo'], cfo_ul=ul['cfo'],
               cfo_dl_coarse=dl['coarse'], cfo_ul_coarse=ul['coarse'],
               cfo_dl_resid=dl['resid'], cfo_ul_resid=ul['resid'],
               snr_dl=dl['snr'], snr_ul=ul['snr'], se_dl=dl['se'], se_ul=ul['se'],
               nslip_dl=dl['nslip'], nslip_ul=ul['nslip'],
               rssi_dl=dl['rssi'], rssi_ul=ul['rssi'],
               clip_dl=dl['clip'], clip_ul=ul['clip'],
               d0_dl=dl['d0'], d0_ul=ul['d0'],
               temp_a=radio.temp(sdr_a) if (sdr_a and rnd % 20 == 0) else np.nan,
               temp_b=radio.temp(sdr_b) if (sdr_b and rnd % 20 == 0) else np.nan)
        lg.add_heavy('H_DL', dl['H']); lg.add_heavy('H_UL', ul['H'])
        if C.LOG_PILOTS:
            lg.add_heavy('Rp_DL', dl['Rp']); lg.add_heavy('Rp_UL', ul['Rp'])
        if rnd <= C.LOG_IQ_N:
            lg.add_heavy('IQ_DL', dl['iq']); lg.add_heavy('IQ_UL', ul['iq'])
        if C.SAVE and C.SAVE_EVERY and rnd % C.SAVE_EVERY == 0:
            lg.checkpoint()

        # Live view only: the half-sum is the cheapest honest summary. Every other
        # estimator is derived offline, where there is one implementation of each.
        fd = (dl['cfo'] + ul['cfo']) / 2
        if rnd % 25 == 0:
            print(HDR)
        print(f"[{rnd:5d}] {rel:+9.2f} {'ST' if static else 'DYN':>4} "
              f"{fd/C.HZ_PER_MPS:+10.3f} {(dl['cfo']-ul['cfo'])/2:+12.1f}   "
              f"{dl['snr']:4.1f}/{ul['snr']:4.1f}")

except KeyboardInterrupt:
    print("\nstopped")
finally:
    n = lg.close() if C.SAVE else len(lg.rows)
    el = (lg.rows[-1]['t_dl'] - T0) if lg.rows else 0.0
    ph = np.array([r['phase'] for r in lg.rows]) if lg.rows else np.zeros(0)
    print(f"\n{n} exchanges in {el:.1f}s ({n/max(el,1e-9):.2f}/s) | "
          f"static {int((ph==0).sum())} dynamic {int((ph==1).sum())} | no-frame {n_bad}")
    print(f"saved -> {PATH}" if C.SAVE else "SAVE=False, nothing written")
    if not C.OFFLINE:
        radio.close(sdr_a, sdr_b)
