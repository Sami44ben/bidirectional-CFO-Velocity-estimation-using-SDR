# matlab/ — the paper figures, in MATLAB

Five scripts that read the recorded captures in `../data/captures/` and write the paper's
figures straight into `../paper/figures/`, each as a vector PDF AND an
editable MATLAB `.fig` next to it. Since 2026-09-28: no tare anywhere, the drift-rate fit
in `icc_load.m` is a TRAILING 3-exchange window (causal), `fig_gap_m` draws the closed form
RMSE^2 = sigma^2 + (r Delta/2)^2 with Delta*, and `fig_density_m` is precision against
pilots and against aperture with the white-noise bound and the model. They reproduce the
Python pipeline; every number below was checked against it.

Run them from this folder (MATLAB R2025a, no toolboxes needed):

```matlab
fig_separation_m     % Fig. 3  -> fig_separation.pdf   ~30 s
fig_gap_m            % Fig. 4  -> fig_gap.pdf          ~1 min
fig_motion_m         % Fig. 5  -> fig_motion.pdf       ~10 s
fig_density_m        % Fig. 6  -> fig_density.pdf      ~8 min (reads a 350 MB capture)
fig_fan_m            % not in the paper -> fig_fan_matlab.pdf
```

Each prints the numbers it plots, so you can paste them into
`results_numbers.tex` or just check them against what is already there.

## The shared file

| file | does |
|---|---|
| `icc_load.m` | loads a capture and adds every derived estimator: half sum, half difference, local drift rate, drift compensation, epoch alignment, the velocities, the SNR gate and the static/dynamic/calibration masks. The MATLAB twin of `../bidir_estimators.py`. |

`icc_load` works from the RAW per-direction fields (`cfo_dl`, `cfo_ul`,
`t_dl_rel`, `t_ul_rel`, `gap_s`), so it does not depend on what the acquisition
script pre-computed and it reads old captures too.

## Which capture each script uses

Change the session stamp at the top of a script to point it somewhere else.

| script | capture |
|---|---|
| `fig_separation_m` | `SEP_0922_0823_*` (20 runs) |
| `fig_gap_m` | `GAP_0922_0840_*` (warm) and `GAP_0922_0912_*` (cold) |
| `fig_motion_m` | `MOVE_REF_test2` |
| `fig_density_m` | `WARM_0922_0807` (needs `Rp_DL`/`Rp_UL`, i.e. `ICC_LOG_PILOTS=1`) |
| `fig_fan_m` | `FAN_OFF1`, `FAN_ON`, `FAN_ON_FAST`, `FAN_OFF2` |

## Verified against the Python pipeline

| quantity | Python | MATLAB |
|---|---|---|
| Doppler gain / LO leakage | 1.0000 ± 0.0001 / 2e-7 | 1.0000 ± 0.0001 / 1.94e-7 |
| separation offset, natural LO | −0.05 Hz, 93 Hz | −0.05 Hz, 93 Hz |
| within-run scatter | 0.030 m/s | 0.030 m/s |
| cold gap 4.1 s: raw → drift comp (trailing W=3, no tare) | 0.542 → 0.068 | 0.542 → 0.068 |
| motion bursts (first three) | 8.42 / 6.56 / 8.96 m | 8.41 / 6.57 / 8.95 m |
| density (std): full comb → PT-RS | 0.0292 → 0.0501 m/s | 0.0292 → 0.0501 m/s |
| fan off / 100 rpm / 800 rpm | 0.088 / 1.23 / 3.97 m/s | 0.088 / 1.23 / 3.97 m/s |

One small, deliberate difference:

* **`fig_fan_m` A–B disagreement** uses `v_align − v_plain` as the proxy, which
  needs nothing beyond this folder; Python computes the two nodes' own
  estimates. Same behaviour, slightly different magnitude (0.58 vs 0.78 at
  100 rpm). Use the Python number if you quote it.

## One MATLAB trap worth remembering

In `slope_hz`, the transposes must be `.'` and never `'`. MATLAB's `'` is the
*conjugate* transpose, so `exp(-1j*s0*T)'` silently becomes `exp(+1j*s0*T)` and
the estimated frequency comes out with the wrong sign. That bug produced a flat
12 m/s floor at every pilot density before it was caught.
