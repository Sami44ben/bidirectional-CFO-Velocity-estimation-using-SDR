"""Capture logging.  Accumulate rows, write a .mat, never block the radio loop.

WHY THE WRITE IS ASYNCHRONOUS.  A synchronous whole-file rewrite every N
exchanges stalled the previous version's acquisition for up to 14 s: 43 of 43
six-sigma outliers in one capture fell in the two exchanges right after a save,
all with the same sign.  The pause let the oscillator walk and the interpolated
estimators read that step as Doppler.  Checkpoints now snapshot and write on a
worker thread, and they carry only the light columns -- the raw pilot and IQ
arrays are written once, at the end.
"""
import os
import threading
import numpy as np
from scipy.io import savemat


class Log:
    def __init__(self, path, meta, heavy=()):
        self.path = path
        self.meta = dict(meta)
        self.rows = []                      # list of dicts, one per exchange
        self.heavy = {k: [] for k in heavy}  # big arrays, final save only
        self._thread = None

    def add(self, **row):
        self.rows.append(row)

    def add_heavy(self, key, arr):
        if key in self.heavy:
            self.heavy[key].append(arr)

    # -- writing ------------------------------------------------------------
    def _pack(self, rows, heavy):
        out = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
               for k in rows[0]}
        out.update(self.meta)
        for k, v in heavy.items():
            if v:
                out[k] = np.array(v)
        return out

    def _write(self, rows, heavy, compress):
        out = self._pack(rows, heavy)
        if compress:
            savemat(self.path, out, do_compression=True)
        else:                               # atomic: a kill mid-write cannot truncate
            tmp = self.path + ".tmp"
            savemat(tmp, out, do_compression=False)
            os.replace(tmp, self.path)

    def checkpoint(self):
        """Non-blocking. Skipped if the previous checkpoint is still writing."""
        if not self.rows or (self._thread and self._thread.is_alive()):
            return
        snap = list(self.rows)              # shallow copy; rows are never mutated
        self._thread = threading.Thread(target=self._write, args=(snap, {}, False),
                                        daemon=True)
        self._thread.start()

    def close(self):
        """Join any checkpoint, then write the authoritative compressed file."""
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=60)
        if self.rows:
            self._write(self.rows, self.heavy, True)
        return len(self.rows)


def header(cfg, path, extra=""):
    """One consistent startup banner for every acquisition script."""
    print(f"\n{path}")
    print(f"  {cfg.FC/1e9:.2f} GHz | {cfg.FS/1e6:.2f} MS/s | {cfg.NSYM} sym "
          f"({cfg.NSYM*cfg.TSYM*1e3:.2f} ms aperture) | {cfg.HZ_PER_MPS:.2f} Hz per m/s")
    if cfg.FD_INJECT or cfg.LO_OFFSET:
        print(f"  injected: Doppler {cfg.FD_INJECT:+.1f} Hz "
              f"({cfg.FD_INJECT/cfg.HZ_PER_MPS:+.2f} m/s) | LO offset {cfg.LO_OFFSET:+.1f} Hz")
    if cfg.STATIC_S > 0:
        print(f"  PHASES: hold still {cfg.STATIC_S:.0f} s "
              f"(t<{cfg.CAL_S:.0f}s calibrates, {cfg.CAL_S:.0f}-{cfg.STATIC_S:.0f}s is the "
              f"held-out floor), then move on the banner.")
    else:
        print("  PHASES: none (whole run marked dynamic)")
    if extra:
        print(extra)
