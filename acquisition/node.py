"""One laptop, one radio.  Run this on BOTH machines, then merge.py.

    ICC_NODE=A python node.py          # laptop A
    ICC_NODE=B python node.py          # laptop B

No link is needed during the run.  Both nodes derive the TDD slot from the wall
clock, so slot k means the same interval on both machines provided the hosts are
NTP-synced; merge.py measures the residual clock offset and reports it.

Node A transmits on even slots, node B on odd, so every exchange is one A->B
burst received by B and one B->A burst received by A.
"""
import os
import time
import numpy as np
import config as C
import dsp
import log as L
import radio

assert C.NODE in ('A', 'B'), "set ICC_NODE=A or ICC_NODE=B"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, "..", "data", "captures"))   # all captures live here
os.makedirs(DATA, exist_ok=True)
PATH = os.path.join(DATA, f"capture_node{C.NODE}_{C.RUN_TAG}.mat")

found = radio.scan()
serial = C.SDR_A if C.NODE == 'A' else C.SDR_B
sdr = radio.open(serial, C.LO_OFFSET if C.NODE == 'B' else 0.0, found)

L.header(C, PATH, extra=f"  node {C.NODE} | slot {C.SLOT_S*1e3:.0f} ms | "
                        f"TX on {'even' if C.NODE == 'A' else 'odd'} slots\n"
                        f"  start BOTH laptops, then wait for BOTH banners before moving")

T0 = time.time()
meta = dict(NODE=C.NODE, SERIAL=serial, RUN_TAG=C.RUN_TAG, NOTE=C.NOTE, T0_EPOCH=T0,
            SLOT_S=C.SLOT_S, FS=C.FS, FC=C.FC, FFTSIZE=C.FFTSIZE, DF=C.DF, NSYM=C.NSYM,
            NPIL=dsp.NPIL, RATIO=C.RATIO, TSYM=C.TSYM, HZ_PER_MPS=C.HZ_PER_MPS,
            FD_INJECT_HW=C.FD_INJECT, LO_OFFSET_HZ=C.LO_OFFSET if C.NODE == 'B' else 0.0,
            TX_GAIN=C.TX_GAIN, RX_GAIN=C.RX_GAIN, GAIN_MODE=C.GAIN_MODE,
            STATIC_S=C.STATIC_S, CAL_FRAC=C.CAL_FRAC, CAL_S=C.CAL_S,
            pilot_active_idx=dsp.PILOT_POS)
lg = L.Log(PATH, meta, heavy=('H', 'Rp', 'IQ'))

HDR = ("\n   exch      t[s]  ph      cfo[Hz]  SNR[dB]  se[Hz]   det%\n  " + "-" * 56)
print(HDR)

tx_even = (C.NODE == 'A')
last_slot, n_rx, n_miss, t_move, txing = -1, 0, 0, None, False
try:
    while not C.MAX_ROUNDS or n_rx < C.MAX_ROUNDS:
        slot = int(time.time() // C.SLOT_S)
        if slot == last_slot:
            time.sleep(0.002)
            continue
        last_slot = slot

        if (slot % 2 == 0) == tx_even:                 # ---- transmit slot ----
            if not txing:
                radio.tx_on(sdr, C.FD_INJECT)
                txing = True
            continue

        if txing:                                       # ---- receive slot ----
            radio.tx_off(sdr)
            txing = False
        sdr.rx()                                        # drop one stale buffer
        t = time.time()
        e = dsp.estimate(np.asarray(sdr.rx()).ravel())
        exch, rel = slot // 2, t - T0

        static = C.STATIC_S > 0 and rel < C.STATIC_S
        if C.STATIC_S > 0 and not static and t_move is None:
            t_move = rel
            print("\n" + "=" * 56)
            print(f"  NODE {C.NODE}: STATIC PHASE DONE ({rel:.1f} s)  ***  MOVE NOW  ***")
            print("=" * 56 + "\n" + HDR)

        if e is None:
            n_miss += 1
            print(f"[{exch:6d}] {rel:+9.2f}  no frame  (det "
                  f"{100*n_rx/max(n_rx+n_miss,1):.0f}%)")
            continue

        n_rx += 1
        lg.add(exch_id=exch, slot=slot, t_host=t, t_rel=rel, phase=0 if static else 1,
               cfo_total=e['cfo'], cfo_coarse=e['coarse'], cfo_resid=e['resid'],
               snr=e['snr'], se_hz=e['se'], nslip=e['nslip'], rssi=e['rssi'],
               clip=e['clip'], d0=e['d0'],
               temp_c=radio.temp(sdr) if n_rx % 20 == 0 else np.nan)
        lg.add_heavy('H', e['H'])
        if C.LOG_PILOTS:
            lg.add_heavy('Rp', e['Rp'])
        if n_rx <= C.LOG_IQ_N:
            lg.add_heavy('IQ', e['iq'])
        if C.SAVE and C.SAVE_EVERY and n_rx % C.SAVE_EVERY == 0:
            lg.checkpoint()

        if n_rx % 25 == 0:
            print(HDR)
        print(f"[{exch:6d}] {rel:+9.2f} {'ST' if static else 'DYN':>4} "
              f"{e['cfo']:+12.1f} {e['snr']:8.1f} {e['se']:7.3f} "
              f"{100*n_rx/(n_rx+n_miss):6.0f}%")

except KeyboardInterrupt:
    print("\nstopped")
finally:
    n = lg.close() if C.SAVE else len(lg.rows)
    print(f"\n{n} bursts received, {n_miss} missed "
          f"(detection {100*n_rx/max(n_rx+n_miss,1):.1f}%)")
    print(f"saved -> {PATH}" if C.SAVE else "SAVE=False, nothing written")
    radio.close(sdr)
