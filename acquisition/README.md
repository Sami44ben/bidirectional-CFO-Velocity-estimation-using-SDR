# acquisition/ — the measurement chain

Drives the two ADALM-Pluto radios and writes one capture per run to `../data/captures/`.
It logs raw per-direction measurements only; every estimator lives in `../analysis/`.

## Files

| file | lines | owns |
|---|---|---|
| `config.py` | ~85 | every knob, for both rigs |
| `dsp.py` | ~180 | waveform + CFO estimator + offline channel |
| `radio.py` | ~95 | Pluto access (the only file importing `adi`) |
| `log.py` | ~90 | row accumulation + non-blocking `.mat` writing |
| `campaign.py` | ~120 | single-host driver: both radios, one laptop |
| `node.py` | ~115 | two-laptop driver: one radio per laptop |
| `merge.py` | ~110 | pair two node captures into one analysable file |
| `selftest.py` | ~60 | no-hardware check that all of the above still works |

## Two rules that keep it small

**One config.** `campaign.py` and `node.py` both `import config`. They cannot run
different numerology. The previous version duplicated the estimator and the
numerology into the node script, and the two drifted to 2.3 vs 3.5 GHz, DF 80 vs
60 kHz, NSYM 140 vs 64 — a divergence that makes the two rigs unable to
demodulate each other and their captures incomparable.

**Acquisition logs raw measurements only.** Per-direction CFO, timestamps, SNR,
fit standard error, phase. It does not compute half-sums, calibrated biases,
time-interpolated or drift-compensated estimates. Those are *estimators*: they
live in `../analysis/bidir_estimators.py`, the analyser applies them, and there is exactly
one implementation of each. The old acquisition script computed ten variants
inline, duplicating the analyser and leaving the question of which one a capture
actually contained.

## Running

```bash
python selftest.py                       # no hardware needed; run after any edit

python campaign.py                       # single host, both radios
ICC_RUN_TAG=FAN2 ICC_MAX_ROUNDS=2000 python campaign.py

ICC_NODE=A python node.py                # laptop A   \  then, on one machine:
ICC_NODE=B python node.py                # laptop B   /  python merge.py

python ../analysis/analyze_icc.py ../data/captures/capture_icc_<TAG>.mat
```

Any UPPERCASE name in `config.py` is overridable as `ICC_<NAME>`, so a sweep never
rewrites the file.

## Phases

Hold still for `STATIC_S`, then move when the banner appears. The first
`CAL_FRAC` of that window is excluded as warm-up; the rest is what the analyser
reports as the error floor. No calibration constant is subtracted anywhere.

On two laptops each node times its window from its own start, so wait for **both**
banners before moving; `merge.py` keeps only the stretch both nodes called static.

## Output

`../data/captures/capture_icc_<TAG>.mat`, in the schema `../analysis/analyze_icc.py` and
`../matlab/icc_load.m` read.

## Deliberately absent

Live plotting, per-direction averaging (`NAVG`), the SNR gate, the ten inline
estimator variants, the field-alias layer, and the separate node numerology.
None of them were load-bearing for the proof of concept. `LOG_PILOTS` and
`LOG_IQ_N` are off by default: pilots cost ~270 kB per exchange and raw IQ ~2 MB,
and neither is needed unless you intend to re-run the estimator from raw samples.
