"""Waveform and CFO estimator.  The only signal processing in the project.

One burst in, one CFO out.  Everything above this file (campaign, node) just
decides WHEN to call it; everything below the estimators (half-sum, calibration,
drift compensation) belongs to the analysis, not here -- the acquisition scripts
log raw per-direction measurements and nothing derived, so there is exactly one
implementation of each estimator and it lives in the analyser.

The frame is built once at import from config, so both rigs transmit the same
waveform by construction.
"""
import numpy as np
import config as C

# --------------------------------------------------------------------------
# Waveform: ZC preamble (two repeats, for timing + coarse CFO) then OFDM payload
# --------------------------------------------------------------------------
_rng = np.random.default_rng(12345)      # fixed: TX and RX must agree on pilots

K0, KNEG = C.FFTSIZE // 2, C.NACT // 2
PILOT_POS = np.arange(1, C.NACT, C.RATIO)
NPIL = PILOT_POS.size
_data_pos = np.setdiff1d(np.arange(C.NACT), PILOT_POS)

_qpsk = lambda *s: (2 * _rng.integers(0, 2, s) - 1 + 1j * (2 * _rng.integers(0, 2, s) - 1)) / np.sqrt(2)
PILOT_SEQ = _qpsk(NPIL)

_grid = np.zeros((C.NACT, C.NSYM), dtype=complex)
_grid[_data_pos, :] = _qpsk(_data_pos.size, C.NSYM)
_grid[PILOT_POS, :] = PILOT_SEQ[:, None]

_fd = np.zeros((C.FFTSIZE, C.NSYM), dtype=complex)
_fd[K0 - KNEG:K0, :] = _grid[:KNEG, :]
_fd[K0 + 1:K0 + 1 + KNEG, :] = _grid[KNEG:, :]
_td = np.fft.ifft(np.fft.ifftshift(_fd, axes=0), axis=0) * np.sqrt(C.FFTSIZE)
_payload = np.vstack([_td[-C.CP:, :], _td]).T.reshape(-1)
_payload /= np.sqrt(np.mean(np.abs(_payload) ** 2))

L_ZC = C.FFTSIZE                          # correlation lag = one ZC repeat
_q = next(q for q in range(2, L_ZC) if np.gcd(q, L_ZC) == 1)
_n = np.arange(L_ZC)
_zc = np.exp(-1j * np.pi * _q * _n * (_n + 1) / L_ZC)
_zc /= np.sqrt(np.mean(np.abs(_zc) ** 2))
_pair = np.concatenate([_zc, _zc])
PREAMBLE = np.concatenate([_pair[-C.CP:], _pair])       # CP + two ZC repeats

_guard = int(round(0.1 * _payload.size))
FRAME = np.concatenate([np.zeros(_guard), PREAMBLE, _payload, np.zeros(_guard)])
FRAME = (FRAME / np.max(np.abs(FRAME)) * 0.8).astype(np.complex64)

NEED = C.NSYM * (C.FFTSIZE + C.CP)                       # payload samples
RXBUF = 1 << int(np.ceil(np.log2(3 * FRAME.size)))
_TX = (FRAME * 2 ** 14).astype(np.complex64)
_tx_n = np.arange(_TX.size)
T_M = np.arange(C.NSYM) * C.TSYM                         # symbol epochs


def tx(fd_hz=0.0):
    """TX samples, optionally pre-rotated by a known Doppler.

    Applied with the SAME sign in both directions: real radial motion does not
    flip with the TX/RX role, which is what makes it separable from the LO."""
    if fd_hz == 0.0:
        return _TX
    return (_TX * np.exp(2j * np.pi * fd_hz * _tx_n / C.FS)).astype(np.complex64)


# --------------------------------------------------------------------------
# Estimator
# --------------------------------------------------------------------------
def _wls(phi, t, w):
    """Weighted LS slope of phi vs t. Returns (slope, weighted mean phi, mean t)."""
    W = w.sum()
    tb = float((w * t).sum() / W)
    pb = float((w * phi).sum() / W)
    den = float((w * (t - tb) ** 2).sum())
    num = float((w * (t - tb) * (phi - pb)).sum())
    return (num / den if abs(den) > 1e-30 else 0.0), pb, tb


def estimate(buf):
    """One received buffer -> measurement dict, or None if no frame was found.

    cfo = coarse + residual.  The residual is measured AFTER derotating by the
    coarse term, so coarse error is absorbed by it and only the SUM is meaningful
    (their correlation is -1 by construction, which is not evidence of accuracy).
    """
    raw = np.asarray(buf).ravel().astype(complex)
    pwr = float(np.mean(np.abs(raw) ** 2))
    rx_dbfs = 10 * np.log10(pwr / 2047.0 ** 2 + 1e-30)          # 12-bit full scale
    clip = float(np.mean((np.abs(raw.real) >= 2040) | (np.abs(raw.imag) >= 2040)))
    r = raw / np.sqrt(pwr + 1e-30)

    # --- detect + coarse CFO from the two repeated ZC halves ---
    ones = np.ones(L_ZC)
    P = np.convolve(np.conj(r[:-L_ZC]) * r[L_ZC:], ones, 'valid')
    Ra = np.convolve(np.abs(r[:-L_ZC]) ** 2, ones, 'valid')
    Rb = np.convolve(np.abs(r[L_ZC:]) ** 2, ones, 'valid')
    n = min(P.size, Ra.size, Rb.size)
    P, Ra, Rb = P[:n], Ra[:n], Rb[:n]
    metric = np.abs(P) ** 2 / (Ra * Rb + 1e-30)
    metric *= (Ra + Rb) > 0.5 * (Ra + Rb).max()                 # ignore the noise floor
    if metric.max() < 0.001:
        return None
    pk = int(np.argmax(metric))
    coarse = float(np.angle(P[pk]) / (2 * np.pi * L_ZC / C.FS))
    r = r * np.exp(-1j * 2 * np.pi * coarse * np.arange(r.size) / C.FS)

    # --- frame timing, then FFT the payload and descramble the pilots ---
    limit = r.size - NEED - PREAMBLE.size
    if limit <= 0:
        return None
    d0 = int(np.argmax(np.abs(np.correlate(r, PREAMBLE, 'valid')[:limit])))
    start = d0 + PREAMBLE.size
    S = r[start:start + NEED].reshape(C.NSYM, C.FFTSIZE + C.CP).T
    Y = np.fft.fftshift(np.fft.fft(S[C.CP:, :], axis=0) / np.sqrt(C.FFTSIZE), axes=0)
    act = np.vstack([Y[K0 - KNEG:K0, :], Y[K0 + 1:K0 + 1 + KNEG, :]])
    Rp = act[PILOT_POS, :] * np.conj(PILOT_SEQ)[:, None]        # (NPIL, NSYM)

    # --- pass 1: rough slope -> channel estimate ---
    phi0 = np.unwrap(np.angle(np.sum(Rp * np.conj(Rp[:, [0]]), axis=0)))
    s0, _, _ = _wls(phi0, T_M, np.ones(C.NSYM))
    H = (Rp * np.exp(-1j * s0 * T_M)[None, :]).mean(axis=1)

    # --- pass 2: per-symbol common phase, matched to H, power-weighted ---
    z = np.sum(Rp * np.conj(H)[:, None], axis=0)
    phi = np.unwrap(np.angle(z))
    w = np.abs(z) ** 2
    s1, p1, t1 = _wls(phi, T_M, w)
    res = phi - (p1 + s1 * (T_M - t1))
    thr = max(0.5 * np.pi, 3.0 * 1.4826 * np.median(np.abs(res - np.median(res))))
    keep = np.abs(res) <= thr                                    # reject cycle slips
    if keep.sum() < 8:                                           # never starve the fit
        keep[:] = True

    slope, pb, tb = _wls(phi[keep], T_M[keep], w[keep])
    resid = slope / (2 * np.pi)

    # Standard error of the slope: a PER-BURST reliability number. It assumes
    # independent per-symbol residuals, so comparing it against the observed
    # run-to-run scatter is a direct test for correlated phase noise.
    ww, rr = w[keep], phi[keep] - (pb + slope * (T_M[keep] - tb))
    ssr = float((ww * rr ** 2).sum())
    den = float((ww * (T_M[keep] - tb) ** 2).sum())
    nk = int(keep.sum())
    se = np.sqrt(ssr / max(nk - 2, 1) / max(den, 1e-30)) / (2 * np.pi) if nk > 2 else np.nan

    # --- pilot EVM as a fit-quality SNR proxy (not calibrated RF SNR) ---
    Rpd = Rp * np.exp(-1j * 2 * np.pi * resid * T_M)[None, :]
    H = Rpd.mean(axis=1)
    snr = 10 * np.log10(np.mean(np.abs(H) ** 2) /
                        max(np.mean(np.abs(Rpd - H[:, None]) ** 2), 1e-30))

    return dict(cfo=coarse + resid, coarse=coarse, resid=resid, snr=float(snr),
                se=float(se), d0=d0, rssi=rx_dbfs, clip=clip,
                nslip=int(C.NSYM - nk), H=H.astype(np.complex64),
                Rp=Rp.astype(np.complex64), iq=raw.astype(np.complex64))


# --------------------------------------------------------------------------
# Offline channel, for testing without radios
# --------------------------------------------------------------------------
_t0 = None
EPS_BV = 8000.0          # emulated natural LO offset (Hz)
DRIFT = -3.0             # Hz/s
SNR_DB = 20.0


def fake_capture(lo_sign, t=None):
    """One synthetic burst. lo_sign=+1 for A->B, -1 for B->A.

    The LO term carries lo_sign and the Doppler term does not: that asymmetry is
    the whole method, so a simulator that got it wrong would silently validate
    nothing."""
    global _t0
    import time
    if _t0 is None:
        _t0 = time.time()
    t = (time.time() - _t0) if t is None else t
    eps = EPS_BV + DRIFT * t - C.LO_OFFSET
    off = C.FD_INJECT + lo_sign * eps
    reps = int(np.ceil(RXBUF / FRAME.size)) + 2
    s = np.tile(FRAME.astype(complex), reps)
    s = np.concatenate([np.zeros(_rng.integers(50, 400)), s])
    s = s * np.exp(2j * np.pi * off * np.arange(s.size) / C.FS)
    amp = np.mean(np.abs(s[np.abs(s) > 1e-6]) ** 2)
    noise = np.sqrt(amp / 10 ** (SNR_DB / 10) / 2) * (
        _rng.standard_normal(s.size) + 1j * _rng.standard_normal(s.size))
    return ((s + noise) * 2047 / np.max(np.abs(s + noise)))[:RXBUF]
