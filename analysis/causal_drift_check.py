import sys, glob, numpy as np
sys.path.insert(0, '.')
from icc_common import load, heldout
from bidir_estimators import _col
W = 21
def trailing(t, f):
    D = np.full(t.size, np.nan)
    for i in range(W-1, t.size):
        s = slice(i-W+1, i+1); ok = np.isfinite(t[s]) & np.isfinite(f[s])
        if ok.sum() > 2: D[i] = np.polyfit(t[s][ok], f[s][ok], 1)[0]
    return D
def rms(x): x = x[np.isfinite(x)]; return np.sqrt(np.mean(x**2))
def run(path, tare_hz=0.0, heldout_only=False):
    d = load(path); hz = float(d['HZ_PER_MPS'])
    t = _col(d, 't_A2B_rel', 't_dl_rel'); gap = _col(d, 'gap_s'); flo = d['f_lo_offset']; fd = d['fd_plain']
    Dc = trailing(np.asarray(t, float), np.asarray(flo, float))
    fdc = fd + Dc*gap/2
    m = heldout(d) if heldout_only else np.ones(fd.size, bool)
    both = m & np.isfinite(d['fd_driftcomp']) & np.isfinite(fdc) & np.isfinite(d['fd_tinterp'])
    f = lambda x: rms((x[both]-tare_hz)/hz)
    return np.nanmedian(gap), f(fd), f(d['fd_driftcomp']), f(fdc), f(d['fd_tinterp'])
print('file                         gap_s  raw   centred causal aligned  [m/s RMSE]')
print('WARM', ['%.3f'%x for x in run('../data/captures/capture_icc_WARM_0922_0807.mat', 0.01, True)])
for tag in ('0840','0912'):
    for p in sorted(glob.glob(f'../data/captures/capture_icc_GAP_0922_{tag}_s*.mat')):
        print(tag, p[-12:-4], ['%.3f'%x for x in run(p, 0.01)])
