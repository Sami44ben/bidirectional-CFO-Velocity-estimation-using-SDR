"""
fan_compare.py -- false peer velocity with the fan off / on / off, radios clamped.

    python fan_compare.py ../data/captures/capture_icc_FAN_OFF1.mat ../data/captures/capture_icc_FAN_ON.mat ../data/captures/capture_icc_FAN_OFF2.mat
    python fan_compare.py OFF1.mat ON_SLOW.mat ON_FAST.mat OFF2.mat --rpm 0,100,800,0 --radius 0.15 --fig ...

Any number of captures; each is labelled from its NOTE ('fan off' / 'fan on ...').
--rpm gives one value per capture (0 = off).

Both radios are clamped in all three captures, so the true peer velocity is
exactly zero throughout and every non-zero output is a measured false velocity.
No tare is applied: the constant bias beta_+ was measured at 0.01 Hz on this pair
with fixed gain, far below the floor.  Reported per capture: RMS and 95th
percentile of the drift-compensated estimate over the held-out window,
the node A / node B disagreement (a reciprocity-loss detector that needs no ground
truth), and the SNR statistics.  --rpm/--radius give the blade-tip speed, which
bounds the scatterer velocity the fan can produce.

Writes <fig>_numbers.json when --fig is given (make_numbers.py reads it).
"""
import os, json, argparse
import numpy as np
from icc_common import load, heldout, masks, metrics, summary
from bidir_estimators import get_variant

EST = 'v_bidir_driftcomp'


def score(d, b0):
    key = EST if EST in d else 'v_bidir_plain'
    v = np.asarray(get_variant(d, key), float) - b0
    # the radios are clamped for the whole capture, so every exchange after the
    # calibration window is at zero true velocity: score all of them, not just
    # the ones the phase flag calls static
    _, _, cal = masks(d)
    ho = ~cal & np.isfinite(v)
    m = metrics(v[ho])
    dis = np.asarray(get_variant(d, 'v_node_disagree'), float) if 'v_node_disagree' in d else np.full(v.size, np.nan)
    dm = metrics(dis[ho])
    s = summary(d)
    return dict(tag=s['tag'], n=m['n'], rms=m['rmse'], p95=m['p95'], p99=m['p99'],
                disagree_rms=dm['rmse'], snr_dl=s['snr_dl'], snr_ul=s['snr_ul'],
                snr_dl_min=float(np.nanmin(get_variant(d, 'snr_at_B'))),
                snr_ul_min=float(np.nanmin(get_variant(d, 'snr_at_A'))),
                dur_s=s['dur_s'], v=v, t=np.asarray(get_variant(d, 't_A2B_rel'), float), ho=ho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+')
    ap.add_argument('--rpm', default=None, help='comma list, one per capture, 0 = off')
    ap.add_argument('--radius', type=float, default=np.nan)
    ap.add_argument('--fig', default=None)
    a = ap.parse_args()
    caps = [load(f) for f in a.files]
    rpms = [float(x) for x in a.rpm.split(',')] if a.rpm else [np.nan] * len(caps)
    labels = []
    for d in caps:
        note = str(d.get('NOTE', '')).strip().lower()
        labels.append('fan off' if 'off' in note or note == '' else note)
    n_off = 0
    for i, lab in enumerate(labels):            # make repeated 'fan off' labels distinct
        if lab == 'fan off':
            n_off += 1
            labels[i] = 'fan off' if n_off == 1 else f'fan off ({n_off})'
    key = EST if EST in caps[0] else 'v_bidir_plain'
    b0 = 0.0                                    # no tare (see docstring)
    res = [score(d, b0) for d in caps]
    print(f"estimator {key}, no tare")
    print(f"  {'':22s} {'n':>5} {'RMS':>7} {'p95':>7} {'p99':>7} {'A-B rms':>8} {'SNR B/A':>10} {'min B/A':>10} {'tip m/s':>8}")
    for lab, r, rpm in zip(labels, res, rpms):
        tip = 2 * np.pi * a.radius * rpm / 60 if np.isfinite(a.radius) and np.isfinite(rpm) and rpm > 0 else np.nan
        r['rpm'] = rpm; r['tip_speed'] = tip
        print(f"  {lab:22s} {r['n']:>5} {r['rms']:>7.3f} {r['p95']:>7.3f} {r['p99']:>7.3f} {r['disagree_rms']:>8.3f} "
              f"{r['snr_dl']:>4.1f}/{r['snr_ul']:<4.1f} {r['snr_dl_min']:>4.1f}/{r['snr_ul_min']:<4.1f} "
              f"{tip if np.isfinite(tip) else float('nan'):>8.1f}")
    out = dict(estimator=key, radius=a.radius,
               runs=[{k: v for k, v in r.items() if k not in ('v', 't', 'ho')} | {'label': lab}
                     for lab, r in zip(labels, res)])
    if a.fig:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'width_ratios': [2, 1]})
        off = 0.0
        for lab, r in zip(labels, res):
            ax[0].plot(r['t'] + off, r['v'], lw=.5, label=lab)
            off += r['t'][-1] + 5
        ax[0].set_xlabel('time, captures concatenated [s]'); ax[0].set_ylabel('false peer velocity [m/s]')
        ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
        for lab, r in zip(labels, res):
            e = np.sort(np.abs(r['v'][r['ho']])); e = e[np.isfinite(e)]
            ax[1].plot(e, np.linspace(0, 1, e.size), label=lab)
        ax[1].set_xlim(left=1e-3)
        ax[1].set_xscale('log'); ax[1].set_xlabel('|false velocity| [m/s]'); ax[1].set_ylabel('CDF')
        ax[1].grid(alpha=.3, which='both'); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(a.fig); plt.close(fig)
        with open(os.path.splitext(a.fig)[0] + '_numbers.json', 'w') as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"  wrote {a.fig}")


if __name__ == '__main__':
    main()
