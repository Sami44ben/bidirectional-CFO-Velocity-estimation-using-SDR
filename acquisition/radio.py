"""Pluto access.  The only file that imports adi/iio.

Everything here is about moving samples; nothing here knows what a CFO is.
Import it lazily -- config.OFFLINE=True must work on a machine with no libiio.
"""
import os
import re
import numpy as np
import config as C
import dsp

# Windows only: where libiio.dll lives.  Set LIBIIO_DIR if it is installed elsewhere.
_DLL_DIRS = (os.environ.get("LIBIIO_DIR", ""),
             r"C:\Program Files\libiio\bin", r"C:\Program Files (x86)\libiio\bin")


def _libiio():
    for d in _DLL_DIRS:
        if os.path.isfile(os.path.join(d, "libiio.dll")) and d not in os.environ["PATH"]:
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


def scan():
    """[(uri, serial)] for every USB Pluto.  Call ONCE, before claiming contexts:
    a serial read after a context is open comes back 'unknown'."""
    _libiio()
    import iio
    out = []
    for uri, desc in iio.scan_contexts().items():
        if uri.startswith("usb:"):
            m = re.search(r"serial=([0-9a-fA-F]+)", desc)
            out.append((uri, m.group(1) if m else "unknown"))
    return out


def open(name, lo_off=0.0, found=None):
    """Open one Pluto by serial or uri and apply the shared configuration."""
    _libiio()
    import adi
    found = scan() if found is None else found
    uri = next((u for u, s in found if name in (u, s)), None)
    if uri is None:
        raise RuntimeError(f"Pluto {name!r} not found. Present: {found}")
    s = adi.Pluto(uri)
    s.sample_rate = C.FS
    s.tx_rf_bandwidth = s.rx_rf_bandwidth = C.FS
    s.tx_lo = s.rx_lo = int(C.FC + lo_off)
    s.tx_hardwaregain_chan0 = C.TX_GAIN
    s.gain_control_mode_chan0 = C.GAIN_MODE
    if C.GAIN_MODE == 'manual':
        s.rx_hardwaregain_chan0 = C.RX_GAIN
    s.rx_buffer_size = dsp.RXBUF
    temp(s)                                    # first read is garbage; prime it
    return s


def temp(sdr):
    """AD936x die temperature, NaN if unavailable or implausible."""
    try:
        t = float(sdr._ctrl.find_channel("temp0").attrs["input"].value) / 1000.0
    except Exception:
        return float('nan')
    return t if 5.0 < t < 120.0 else float('nan')


def burst(tx_sdr, rx_sdr, fd=0.0):
    """One directional capture: gate TX on, receive, gate TX off.

    The LO is never retuned, so no synthesizer settling enters the measurement."""
    tx_sdr.tx_destroy_buffer()
    tx_sdr.tx_cyclic_buffer = True
    tx_sdr.tx(dsp.tx(fd).copy())
    for _ in range(C.RX_FLUSH):
        rx_sdr.rx()
    r = np.asarray(rx_sdr.rx()).ravel()
    tx_sdr.tx_destroy_buffer()
    return r


def tx_on(sdr, fd=0.0):
    sdr.tx_destroy_buffer()
    sdr.tx_cyclic_buffer = True
    sdr.tx(dsp.tx(fd).copy())


def tx_off(sdr):
    sdr.tx_destroy_buffer()


def close(*sdrs):
    """Destroy BOTH buffers before the objects are collected: a live RX buffer is
    what makes libiio raise an access violation at interpreter shutdown."""
    for s in sdrs:
        for m in ('tx_destroy_buffer', 'rx_destroy_buffer'):
            try:
                getattr(s, m)()
            except Exception:
                pass
