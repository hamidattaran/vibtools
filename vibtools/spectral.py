"""Spectral estimation and test-data helpers (PSD, FRF, coherence, profiles)."""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid


def _nperseg(fs, df, n):
    return int(min(round(fs / df), n))


def psd_welch(x, fs, df=1.0, window="hann", overlap=0.5, detrend="constant"):
    """One-sided PSD by Welch's method.

    ``df`` is the requested frequency resolution in Hz (the actual value is
    fs / nperseg). If ``x`` is in g the PSD is in g^2/Hz.
    """
    x = np.asarray(x, float)
    n = _nperseg(fs, df, x.size)
    return signal.welch(x, fs=fs, window=window, nperseg=n,
                        noverlap=int(n * overlap), detrend=detrend,
                        scaling="density")


def grms(f, psd, f1=None, f2=None):
    """RMS value from a PSD over [f1, f2] (trapezoidal area under the PSD).

    Always quote the band you used. Bins are not split at f1 and f2.
    """
    f = np.asarray(f, float)
    psd = np.asarray(psd, float)
    lo = f[0] if f1 is None else f1
    hi = f[-1] if f2 is None else f2
    m = (f >= lo) & (f <= hi)
    return float(np.sqrt(trapezoid(psd[m], f[m])))


def frf_estimates(x_in, y_out, fs, df=1.0, window="hann", overlap=0.5):
    """H1, H2 and coherence between an input and an output signal.

    H1 = Pxy / Pxx  (use when noise is mostly on the output)
    H2 = Pyy / Pyx  (use when noise is mostly on the input)
    Returns f, H1, H2, coherence.
    """
    x_in = np.asarray(x_in, float)
    y_out = np.asarray(y_out, float)
    n = _nperseg(fs, df, min(x_in.size, y_out.size))
    kw = dict(fs=fs, window=window, nperseg=n, noverlap=int(n * overlap))
    f, Pxx = signal.welch(x_in, **kw)
    _, Pyy = signal.welch(y_out, **kw)
    _, Pxy = signal.csd(x_in, y_out, **kw)
    with np.errstate(divide="ignore", invalid="ignore"):
        H1 = Pxy / Pxx
        H2 = Pyy / np.conj(Pxy)
        coh = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    return f, H1, H2, coh


def asd_from_breakpoints(breakpoints, f):
    """Evaluate a test-spec ASD on frequencies ``f`` (log-log interpolation).

    breakpoints: [(f1, asd1), (f2, asd2), ...] in Hz and g^2/Hz. The value is
    zero outside the first and last breakpoint.
    """
    bp = np.asarray(breakpoints, float)
    f = np.asarray(f, float)
    out = np.zeros_like(f)
    m = (f >= bp[0, 0]) & (f <= bp[-1, 0])
    out[m] = np.exp(np.interp(np.log(f[m]), np.log(bp[:, 0]), np.log(bp[:, 1])))
    return out


def velocity_from_asd(f, asd, f1=None, f2=None, g=9.81):
    """Velocity and displacement RMS from an acceleration ASD in g^2/Hz.

    Divides the acceleration PSD by (2 pi f)^2 for velocity and (2 pi f)^4 for
    displacement, then integrates over [f1, f2]. Use the INPUT (control) ASD to
    get the shaker table motion. For a response ASD this gives the absolute
    motion at that sensor; to recover the input from a response first divide by
    |T(f)|^2. Random motion has no initial velocity: these are 1-sigma values
    of a zero-mean process. Low frequencies dominate, so state the band.

    Returns dict with grms [g], v_rms [m/s], d_rms [m], and the frequency,
    velocity PSD [(m/s)^2/Hz] and displacement PSD [m^2/Hz] arrays used.
    """
    f = np.asarray(f, float)
    asd = np.asarray(asd, float)
    lo = f[0] if f1 is None else f1
    hi = f[-1] if f2 is None else f2
    m = (f >= lo) & (f <= hi) & (f > 0)
    fb, ab = f[m], asd[m] * g ** 2
    w2 = (2 * np.pi * fb) ** 2
    v_psd, d_psd = ab / w2, ab / w2 ** 2
    return dict(grms=float(np.sqrt(trapezoid(asd[m], fb))),
                v_rms=float(np.sqrt(trapezoid(v_psd, fb))),
                d_rms=float(np.sqrt(trapezoid(d_psd, fb))),
                f=fb, v_psd=v_psd, d_psd=d_psd)


def grms_from_breakpoints(breakpoints, n=20000):
    """Grms of a breakpoint ASD profile."""
    bp = np.asarray(breakpoints, float)
    f = np.linspace(bp[0, 0], bp[-1, 0], n)
    return grms(f, asd_from_breakpoints(bp, f))


def synth_gaussian(f, G, duration, fs, seed=None):
    """Synthesise a Gaussian time history with a target one-sided PSD.

    Random-phase spectral method; the record repeats with period ``duration``.
    Returns (t, x). Useful for self-checks and Monte-Carlo fatigue comparisons.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * fs))
    fk = np.arange(n // 2 + 1) * (fs / n)
    Gk = np.interp(fk, np.asarray(f, float), np.asarray(G, float), left=0.0, right=0.0)
    a = np.sqrt(2.0 * Gk * (fs / n))
    c = (n / 2.0) * a * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, fk.size))
    c[0] = 0.0
    if n % 2 == 0:
        c[-1] = 0.0
    return np.arange(n) / fs, np.fft.irfft(c, n=n)


def campbell_orders(rpm, orders):
    """Frequencies (Hz) of excitation orders versus speed, for a Campbell diagram.

    rpm: array of speeds. orders: iterable of orders (1 = rotation frequency;
    gear mesh = number of teeth). Returns {order: freq_array}.
    """
    rpm = np.asarray(rpm, float)
    return {o: o * rpm / 60.0 for o in orders}
