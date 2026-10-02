"""No-hardware check that the DSP and the acquisition chain still work.

    python selftest.py

Run this after ANY edit to dsp.py or config.py.  It fails loudly rather than
printing something reassuring: the point is to catch a numerology or estimator
change that would otherwise only show up as bad data after a measurement session.
"""
import subprocess
import sys
import numpy as np
import config as C
import dsp

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}   {detail}")
    if not ok:
        fails.append(name)


print(f"\nnumerology: {C.FC/1e9:.2f} GHz | {C.FS/1e6:.2f} MS/s | NSYM {C.NSYM} "
      f"| aperture {C.NSYM*C.TSYM*1e3:.2f} ms | NPIL {dsp.NPIL} | RXBUF {dsp.RXBUF}")

# 1. the estimator must return an injected offset, over the whole coarse range
rng = np.random.default_rng(0)
errs = []
for true in (0.0, 5.0, -12.5, 100.0, -1000.0, 8000.0, -20000.0):
    reps = int(np.ceil(dsp.RXBUF / dsp.FRAME.size)) + 2
    s = np.concatenate([np.zeros(137), np.tile(dsp.FRAME.astype(complex), reps)])
    s = s * np.exp(2j * np.pi * true * np.arange(s.size) / C.FS)
    a = np.mean(np.abs(s[np.abs(s) > 1e-6]) ** 2)
    s = s + np.sqrt(a / 10 ** 2.5 / 2) * (rng.standard_normal(s.size)
                                          + 1j * rng.standard_normal(s.size))
    e = dsp.estimate((s * 2047 / np.max(np.abs(s)))[:dsp.RXBUF])
    errs.append(abs(e['cfo'] - true) if e else np.inf)
check("CFO recovery over +-20 kHz", max(errs) < 1.0, f"worst |err| {max(errs):.3f} Hz")

# 2. the sign convention: Doppler must NOT flip with direction, the LO must.
#    Getting this backwards would make drift look like motion, so it is the one
#    property worth asserting rather than trusting.
C.FD_INJECT, C.LO_OFFSET, dsp._t0 = 50.0, 0.0, None
dl = dsp.estimate(dsp.fake_capture(+1, t=0.0))
ul = dsp.estimate(dsp.fake_capture(-1, t=0.0))
half, diff = (dl['cfo'] + ul['cfo']) / 2, (dl['cfo'] - ul['cfo']) / 2
check("half-sum recovers injected Doppler", abs(half - 50.0) < 1.0,
      f"{half:+.2f} Hz vs +50.00 injected")
check("half-difference carries the LO term", abs(diff - dsp.EPS_BV) < 1.0,
      f"{diff:+.1f} Hz vs {dsp.EPS_BV:+.1f} emulated")
C.FD_INJECT = 0.0

# 3. the whole acquisition path, offline, end to end
r = subprocess.run([sys.executable, "campaign.py"], capture_output=True, text=True,
                   env={**__import__('os').environ, "ICC_OFFLINE": "1",
                        "ICC_RUN_TAG": "SELFTEST", "ICC_MAX_ROUNDS": "40",
                        "ICC_STATIC_S": "4"})
ok = r.returncode == 0 and "exchanges in" in r.stdout
check("campaign.py runs offline", ok, r.stdout.strip().splitlines()[-2] if ok else r.stderr[-200:])

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {', '.join(fails)}"))
sys.exit(1 if fails else 0)
