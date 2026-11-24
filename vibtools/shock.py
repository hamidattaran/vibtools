"""Shock response spectrum (SRS) and pulse helpers.

The SRS uses a ramp-invariant digital filter (Smallwood) for an SDOF oscillator
on a moving base. Results are maximax values (largest absolute response, with
zero padding to capture residual response after the pulse ends).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import lfilter

from .sdof import ramp_invariant_coeffs


@dataclass
class SRSResult:
    freqs: np.ndarray
    Q: float
    acc_pos: np.ndarray      # positive peak absolute acceleration
    acc_neg: np.ndarray      # negative peak (as a positive number)
    acc_max: np.ndarray      # maximax absolute acceleration
    rel_disp: np.ndarray     # maximax relative displacement
    pseudo_vel: np.ndarray   # 2*pi*f * rel_disp


def octave_freqs(f1, f2, n_per_oct=12):
    """Frequencies from f1 up to f2 with ``n_per_oct`` points per octave."""
    n = int(np.floor(n_per_oct * np.log2(f2 / f1) + 1e-9)) + 1
    return f1 * 2.0 ** (np.arange(n) / n_per_oct)


def srs(accel, fs, freqs, Q=10.0, residual=True):
    """Shock response spectrum of a base acceleration time history.

    Units: ``accel`` in any unit U gives acceleration SRS in U. Relative
    displacement is in U/(rad/s)^2, so use m/s^2 to get metres and pseudo
    velocity in m/s.
    """
    x = np.asarray(accel, float)
    freqs = np.asarray(freqs, float)
    if freqs.max() > fs / 8.0:
        warnings.warn(f"max SRS frequency {freqs.max():.0f} Hz exceeds fs/8 "
                      f"({fs / 8:.0f} Hz); accuracy degrades. Resample faster.")
    if residual:
        pad = int(np.ceil(2.0 * fs / freqs.min()))
        x = np.concatenate([x, np.zeros(pad)])
    pos = np.empty(freqs.size)
    neg = np.empty(freqs.size)
    rd = np.empty(freqs.size)
    for i, f in enumerate(freqs):
        b, a = ramp_invariant_coeffs(f, Q, fs, "abs_acc")
        y = lfilter(b, a, x)
        pos[i], neg[i] = y.max(), -y.min()
        b, a = ramp_invariant_coeffs(f, Q, fs, "rel_disp")
        rd[i] = np.abs(lfilter(b, a, x)).max()
    return SRSResult(freqs, Q, pos, neg, np.maximum(pos, neg), rd,
                     2.0 * np.pi * freqs * rd)


def half_sine(peak, duration, fs, lead=0.0, tail=0.0):
    """Half-sine pulse (peak in any unit, duration in s). Returns (t, a)."""
    n = int(round(duration * fs)) + 1
    t_p = np.arange(n) / fs
    a_p = peak * np.sin(np.pi * t_p / duration)
    a = np.concatenate([np.zeros(int(lead * fs)), a_p, np.zeros(int(tail * fs))])
    return np.arange(a.size) / fs, a


def velocity_change(accel, fs):
    """Net velocity change of an acceleration pulse (area under a(t))."""
    return float(trapezoid(np.asarray(accel, float), dx=1.0 / fs))


def srs_compare(spec_a, spec_b, tol_db=3.0):
    """dB difference 20*log10(a/b) and a within-tolerance flag per frequency."""
    a = np.asarray(spec_a, float)
    b = np.asarray(spec_b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(a / b)
    return db, np.abs(db) <= tol_db
