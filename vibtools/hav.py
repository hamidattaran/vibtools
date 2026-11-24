"""Hand-arm vibration (HAV): W_h frequency weighting, vibration total value and A(8).

The W_h weighting follows the filter definition used by ISO 8041 for ISO 5349-1:
a band-limiting high-pass (f1 = 6.310 Hz) and low-pass (f2 = 1258.9 Hz), both
second-order Butterworth, times an acceleration-velocity transition
(f3 = f4 = 15.915 Hz, Q4 = 0.64). The magnitude reproduces the one-third-octave
weighting factors of ISO 5349-1 Annex A within 0.5 % (see the tests). Verify
against the standard edition your lab uses before relying on it for declarations.

Signals are weighted in the frequency domain (FFT), which applies the exact
analog magnitude and phase. Use records much longer than 1/6 Hz (several seconds).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

F1, F2, F3, F4, Q4 = 6.310, 1258.9, 15.915, 15.915, 0.64


def wh_response(f):
    """Complex W_h frequency response at frequencies f (Hz)."""
    f = np.asarray(f, float)
    s = 2j * np.pi * np.where(f == 0, 1e-12, f)
    w1, w2, w3, w4 = (2 * np.pi * x for x in (F1, F2, F3, F4))
    hp = 1 / (1 + np.sqrt(2) * w1 / s + (w1 / s) ** 2)
    lp = 1 / (1 + np.sqrt(2) * s / w2 + (s / w2) ** 2)
    tr = (1 + s / w3) / (1 + s / (Q4 * w4) + (s / w4) ** 2)
    h = hp * lp * tr
    return np.where(f == 0, 0.0, h)


def wh_weight(x, fs):
    """Apply W_h to an acceleration time history (any unit; output in the same unit)."""
    x = np.asarray(x, float)
    n = x.size
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(n, 1 / fs)
    return np.fft.irfft(X * wh_response(f), n)


def weighted_rms(x, fs):
    """Frequency-weighted RMS acceleration a_hw of one axis."""
    return float(np.sqrt(np.mean(wh_weight(x, fs) ** 2)))


def weighted_rms_from_asd(f, asd):
    """a_hw from a one-sided acceleration ASD: sqrt(integral |W_h|^2 G df).

    Units follow the ASD: (m/s^2)^2/Hz gives m/s^2. Useful for FEA harmonic or
    random results at the handle.
    """
    f = np.asarray(f, float)
    g = np.abs(wh_response(f)) ** 2 * np.asarray(asd, float)
    return float(np.sqrt(trapezoid(g, f)))


def vibration_total_value(ax, ay=None, az=None, fs=None):
    """a_hv = sqrt(a_hwx^2 + a_hwy^2 + a_hwz^2).

    Pass three time histories with fs, or three already-weighted RMS values.
    """
    vals = [v for v in (ax, ay, az) if v is not None]
    if fs is not None:
        vals = [weighted_rms(v, fs) for v in vals]
    return float(np.sqrt(sum(float(v) ** 2 for v in vals)))


def daily_exposure(a_hv, hours):
    """A(8) in m/s^2 for one or several tasks: sqrt(sum(a_i^2 T_i) / 8 h).

    a_hv and hours may be scalars or equal-length sequences (one per task).
    """
    a = np.atleast_1d(np.asarray(a_hv, float))
    t = np.atleast_1d(np.asarray(hours, float))
    return float(np.sqrt(np.sum(a ** 2 * t) / 8.0))


def time_to_reach(a_hv, a8_target):
    """Hours of exposure at a_hv that give A(8) = a8_target (e.g. 2.5 or 5 m/s^2)."""
    return float(8.0 * (a8_target / a_hv) ** 2)
