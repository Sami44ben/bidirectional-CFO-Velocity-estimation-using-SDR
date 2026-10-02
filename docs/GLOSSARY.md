# What every name in the capture files means

Generated reference for `capture_*.mat`. Old names still work everywhere — new
self-describing names are written **alongside** them, nothing was renamed away.

Run `python bidir_estimators.py` to print this from the code itself.

## The measurement

One exchange is two bursts, in this order on the wire:

```
DL[k]   A transmits, B receives  ->  B measures cfo_A2B[k] at t_A2B[k]
UL[k]   B transmits, A receives  ->  A measures cfo_B2A[k] at t_B2A[k]
```

Each radio measures exactly **one** number:

```
cfo_A2B = f_d + f_lo        (measured at B)
cfo_B2A = f_d - f_lo        (measured at A)
```

`f_d` is Doppler — **same sign both ways**, because physical motion doesn't care
which radio is transmitting. `f_lo` is the LO offset — **opposite sign**, because
swapping the TX/RX role swaps which oscillator gets subtracted. That asymmetry is
the whole method:

```
f_d   = (cfo_A2B + cfo_B2A)/2      half-SUM   kills the oscillator
f_lo  = (cfo_A2B - cfo_B2A)/2      half-DIFF  keeps only the oscillator
```

**Neither node can do this alone.** A only ever holds `cfo_B2A`, B only ever holds
`cfo_A2B`. One number must cross the link before any velocity exists. That is the
structure of the problem, not an implementation shortcut — and it's why Jensen &
Bokulic telemeter their counter values to the ground rather than solving it onboard.

## Raw per-direction measurements

| Old name | New name | Meaning |
|---|---|---|
| `cfo_dl` | `cfo_A2B` | CFO of the A→B burst, **measured at node B** [Hz] |
| `cfo_ul` | `cfo_B2A` | CFO of the B→A burst, **measured at node A** [Hz] |
| `cfo_dl_coarse` | `cfo_A2B_coarse` | coarse (preamble) part of `cfo_A2B` [Hz] |
| `cfo_ul_coarse` | `cfo_B2A_coarse` | coarse (preamble) part of `cfo_B2A` [Hz] |
| `cfo_dl_resid` | `cfo_A2B_resid` | residual (pilot-slope) part of `cfo_A2B` [Hz] |
| `cfo_ul_resid` | `cfo_B2A_resid` | residual (pilot-slope) part of `cfo_B2A` [Hz] |
| `t_dl` | `t_A2B` | epoch B measured the A→B burst [s] |
| `t_ul` | `t_B2A` | epoch A measured the B→A burst [s] |
| `gap_s` | `gap_s` | `t_B2A − t_A2B`, the DL→UL measurement gap [s] |
| `snr_dl` | `snr_at_B` | pilot-EVM SNR of the burst **B** received [dB] |
| `snr_ul` | `snr_at_A` | pilot-EVM SNR of the burst **A** received [dB] |
| `se_dl` | `se_at_B` | standard error of B's CFO slope fit [Hz] |
| `se_ul` | `se_at_A` | standard error of A's CFO slope fit [Hz] |

## Fused quantities

| Old name | New name | Meaning |
|---|---|---|
| `eps_bv` | `f_lo_offset` | LO offset between the pair, `(A2B − B2A)/2` [Hz] |
| `fd_half` | `fd_plain` | half-sum Doppler, no correction [Hz] |
| `fd_cal` | `fd_biascal` | half-sum minus a **static** bias constant [Hz] |
| `fd_ta` | `fd_tinterp` | A→B interpolated to the B→A epoch, then half-sum [Hz] |
| `b_cal` | `bias_plain_hz` | the static constant `fd_biascal` subtracts [Hz] |
| `b_cal_ta` | `bias_tinterp_hz` | the static constant `fd_tinterp` subtracts [Hz] |

## The estimators, worst to best

| Name | What it does | Static RMSE |
|---|---|---|
| `v_oneway_nocorr` | **one direction's TOTAL CFO read as if it were Doppler.** No frequency correction of any kind — so it reports the whole LO offset (thousands of Hz) as velocity. This is the strawman baseline, *not* a real receiver | 154.28 m/s |
| `v_oneway_afc` | one-way after a **one-time AFC lock** at t=0. Fairer, but drift-limited | 486.06 m/s |
| `v_bidir_plain` | the half-sum. No correction | 0.0987 m/s |
| `v_bidir_backpair` | UL[k] paired with the *following* DL[k+1] instead of the preceding one | 0.5307 m/s |
| `v_bidir_sympair` | mean of the forward and backward pairings — drift artefacts have opposite sign and partly cancel | 0.2570 m/s |
| `v_bidir_biascal` | half-sum minus a static bias measured on a calibration run | 0.2930 m/s |
| `v_bidir_driftcomp` | half-sum `+ D·g/2`. **No calibration at all**, and uses no interpolation, so it keeps every exchange | 0.0712 m/s |
| `v_bidir_tinterp` | DL interpolated to the UL epoch (= `v_nodeA`) | **0.0657 m/s** |
| `v_nodeA` | **node A's own estimate**, anchored at A's epoch `t_B2A` | **0.0657 m/s** |
| `v_nodeB` | **node B's own estimate**, anchored at B's epoch `t_A2B` | **0.0656 m/s** |
| `v_node_disagree` | `v_nodeA − v_nodeB`. Needs **no ground truth** | — |

The three interpolating estimators drop ~12% of exchanges (the ones adjacent to a
save stall — see below) and win on the rest. `v_bidir_driftcomp` keeps every
exchange and is 8% worse. Which you prefer depends on whether you would rather
have the tighter number or the complete series.

## Why `v_1way` looks absurd, and what to say about it

`v_1way` (now `v_oneway_nocorr`) is literally `cfo_A2B / HZ_PER_MPS`. It divides the
**entire** measured CFO of one direction — which is ~99.99% LO offset — by
7.67 Hz/(m/s). In the static run `cfo_A2B[0] = +4291 Hz`, of which `f_d = 0` and
`f_lo = +4288 Hz`, so it reports **+559.7 m/s**.

It is *not* a residual after correction. It is the uncorrected total.

Quoting "63.9 dB of cancellation" against that baseline will draw fire, because no
real one-way receiver runs without frequency correction. The honest framing:

- one-way, no correction → 154 m/s
- one-way, one-time AFC → 486 m/s (drift-limited, *worse*, since `f_lo` walks away)
- one-way, continuous AFC → ~3.4 m/s **but it also nulls any real Doppler**

That last line is the actual argument. **A one-way link is *unidentifiable*, not
merely inaccurate**: the Doppler and the LO term are perfectly confounded, and any
loop that tracks out oscillator drift tracks out the velocity with it. The
bidirectional exchange breaks the degeneracy. That is a structural claim and it is
much stronger than a large dB number.

## Drift compensation — why it needs no calibration

The two bursts aren't simultaneous; UL happens a gap `g` after DL. Carrying the
drift through:

```
fd_plain = f_d + [f_lo(t_A2B) − f_lo(t_B2A)]/2 = f_d − D·g/2
```

where `D = d f_lo/dt`. So the half-sum's bias is **not an unknown hardware
constant** — it is the drift rate times half the gap, and *both are measured every
exchange* (`D` from the local slope of `f_lo_offset`, `g` from `gap_s`). Adding
`D·g/2` back needs no calibration run, no static pass, nothing pasted between runs.

It also explains why `b_cal` goes stale: `b_cal ≈ D·g/2`, and in the lab data `D`
decayed from −24.3 to −2.5 Hz/s inside one 14-minute run. A constant fitted at the
start is ~10× wrong by the end, which is exactly why `v_bidir_biascal` (0.293) is
*worse* than doing nothing (0.0987).

## The per-node consistency check

`v_nodeA` and `v_nodeB` are the same physical quantity computed by two different
radios, each anchored at its own measurement epoch. They must agree. How far they
disagree measures the loss of reciprocity between the DL and UL epochs:

```
                     mean(v_A − v_B)      rms      p95
static run              −0.0003 m/s     0.0421   0.0831
fan run                 −0.0116 m/s     1.2671   2.4725     <- 30x worse
```

The means agree to sub-mm/s — the estimators are unbiased relative to each other.
But the **rms disagreement is 30x larger when the fan runs**, with no ground truth
involved anywhere. That makes it an integrity monitor: it detects the multipath
breakdown from the data alone.

## The save-stall artefact (found via the live plot)

The campaign pauses to rewrite the whole capture every `SAVE_EVERY` exchanges. On
the lab runs that pause is **2.0 s early and 14.7 s by the end** (the file grows),
and 100% of the pauses land exactly on a multiple of 25 = `SAVE_EVERY`.

During the pause the radios stop transmitting. The first exchange afterwards comes
back with the LO tens of Hz off where its neighbours sit — a measured **44 Hz step
across one 0.49 s interval**, against 2.3 Hz of ordinary drift. Its own half-sum is
fine, because both directions moved together. But any estimator that *interpolates*
between it and its neighbour amplifies that step into a ~1.5 m/s spike.

That single artefact was costing the interpolating estimators a factor of four:

```
                   before guard   after guard
v_bidir_tinterp       0.2432        0.0657
v_nodeA               0.2550        0.0657
v_nodeB               0.2528        0.0656
node-agreement rms    0.3544        0.0421
```

`bidir_estimators.py` now refuses to interpolate across such a hole, and masks the
exchanges on either side of one. The two nodes need **opposite** masks — node A at
k spans {k, k+1} while node B spans {k−1, k} — so a single symmetric mask fixes A
and leaves B untouched.

The real fix is to stop stalling the loop: log `Rp` only when you need it, raise
`SAVE_EVERY`, or write incrementally instead of rewriting 278 MB each time.

## Caveat on recomputation

The campaign script logs `v_nodeA` / `v_nodeB` / `v_bidir_driftcomp` online using a
**trailing** drift window (it can't see the future). `analyze_icc.py` recomputes
them from the raw CFOs with a **centred** window, which is slightly better, and
overwrites the logged values. Small differences between the live print and the
analysis table are expected and the analysis is authoritative.
