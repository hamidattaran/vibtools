"""Fatigue for harmonic, sine-sweep, random and time-history loading.

S-N convention
--------------
``SN(C, m)``: N * Sa**m = C, where Sa is the stress AMPLITUDE (half the range)
and N the cycles to failure. There is no endurance limit (conservative for
high-cycle work). All stress inputs must share one unit (for example MPa);
stress PSDs are in unit^2/Hz.

Damage D is the Miner sum; failure is predicted at D = 1. Always report a
range, because D scales with stress^m.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import gamma as G_


@dataclass(frozen=True)
class SN:
    C: float
    m: float

    def cycles(self, sa):
        return self.C / np.power(np.asarray(sa, float), self.m)

    @classmethod
    def from_points(cls, p1, p2):
        """Straight line in log-log through (Sa1, N1) and (Sa2, N2)."""
        (s1, n1), (s2, n2) = p1, p2
        m = np.log(n2 / n1) / np.log(s1 / s2)
        return cls(C=float(n1 * s1**m), m=float(m))

    @classmethod
    def from_basquin(cls, sigma_f, b):
        """Basquin form Sa = sigma_f * (2N)^b, with b < 0."""
        m = 1.0 / abs(b)
        return cls(C=float(0.5 * sigma_f**m), m=float(m))


# --------------------------------------------------------------------------
# Mean stress and Miner
# --------------------------------------------------------------------------
def goodman(sa, sm, Su):
    """Equivalent fully reversed amplitude, Goodman. Compressive mean is ignored."""
    sm = np.maximum(np.asarray(sm, float), 0.0)
    return np.where(sm < Su, np.asarray(sa, float) / (1.0 - sm / Su), np.inf)


def gerber(sa, sm, Su):
    """Equivalent fully reversed amplitude, Gerber. Compressive mean is ignored."""
    sm = np.maximum(np.asarray(sm, float), 0.0)
    return np.where(sm < Su, np.asarray(sa, float) / (1.0 - (sm / Su) ** 2), np.inf)


def miner(blocks, sn, Su=None):
    """Miner damage for blocks of constant-amplitude cycles.

    blocks: iterable of (sa, n_cycles) or (sa, n_cycles, sm). Goodman is applied
    when ``Su`` is given and a mean stress is supplied.
    """
    D = 0.0
    for b in blocks:
        sa, n = b[0], b[1]
        if Su is not None and len(b) > 2:
            sa = float(goodman(sa, b[2], Su))
        D += n / float(sn.cycles(sa))
    return D


# --------------------------------------------------------------------------
# Sine: dwell and sweep
# --------------------------------------------------------------------------
def dwell_damage(sa, f, seconds, sn):
    """Damage from a sine dwell: cycles = f * t."""
    return f * seconds / float(sn.cycles(sa))


def log_sweep_cycles_per_hz(rate_oct_per_min):
    """Cycles accumulated per Hz for a logarithmic sweep (constant in frequency)."""
    return 60.0 / (rate_oct_per_min * np.log(2.0))


def sweep_damage(f, sa, sn, rate, kind="log", n_sweeps=1):
    """Damage accumulated over a sine sweep.

    f, sa : frequency grid (Hz) and stress amplitude at each frequency, for
        example from a harmonic response run at the specified input level.
    rate  : octaves/min for kind='log'; Hz/s for kind='linear'.
    Integrates cycle density / allowable cycles over frequency.
    """
    f = np.asarray(f, float)
    if kind == "log":
        density = np.full_like(f, log_sweep_cycles_per_hz(rate))
    elif kind == "linear":
        density = f / rate
    else:
        raise ValueError("kind must be 'log' or 'linear'")
    return n_sweeps * float(trapezoid(density / sn.cycles(sa), f))


# --------------------------------------------------------------------------
# Random: spectral methods
# --------------------------------------------------------------------------
def spectral_moments(f, Gs, orders=(0, 1, 2, 4)):
    """Spectral moments m_n = integral of f^n G(f) df of a one-sided stress PSD."""
    f = np.asarray(f, float)
    Gs = np.asarray(Gs, float)
    return {n: float(trapezoid(f**n * Gs, f)) for n in orders}


def rate_of_peaks(f, Gs):
    m = spectral_moments(f, Gs)
    return float(np.sqrt(m[4] / m[2]))


def rate_of_zero_crossings(f, Gs):
    m = spectral_moments(f, Gs)
    return float(np.sqrt(m[2] / m[0]))


def narrowband_damage(f, Gs, T, sn, rate="peaks"):
    """Rayleigh-amplitude damage (narrow-band, Gaussian). Time T in seconds."""
    m = spectral_moments(f, Gs)
    nu = np.sqrt(m[4] / m[2]) if rate == "peaks" else np.sqrt(m[2] / m[0])
    return float(nu * T / sn.C * 2 ** (sn.m / 2) * m[0] ** (sn.m / 2) * G_(1 + sn.m / 2))


def dirlik_parameters(m0, m1, m2, m4):
    xm = m1 / m0 * np.sqrt(m2 / m4)
    g = m2 / np.sqrt(m0 * m4)
    D1 = 2.0 * (xm - g * g) / (1.0 + g * g)
    R = (g - xm - D1**2) / (1.0 - g - D1 + D1**2)
    D2 = (1.0 - g - D1 + D1**2) / (1.0 - R)
    D3 = 1.0 - D1 - D2
    Q = 1.25 * (g - D3 - D2 * R) / D1 if abs(D1) > 1e-12 else np.inf
    return dict(D1=D1, D2=D2, D3=D3, R=R, Q=Q, xm=xm, gamma=g)


def dirlik_damage(f, Gs, T, sn):
    """Dirlik wide-band damage from a stress PSD. Time T in seconds.

    Returns dict(D, life_s, rms, EP, params). life_s = T / D.
    """
    mom = spectral_moments(f, Gs)
    m0, m1, m2, m4 = mom[0], mom[1], mom[2], mom[4]
    p = dirlik_parameters(m0, m1, m2, m4)   # R enters the PDF squared, so |R| is used
    m = sn.m
    term1 = 0.0 if not np.isfinite(p["Q"]) else p["D1"] * p["Q"] ** m * G_(1 + m)
    integral = term1 + 2 ** (m / 2) * G_(1 + m / 2) * (p["D2"] * abs(p["R"]) ** m + p["D3"])
    EP = np.sqrt(m4 / m2)
    D = float(EP * T * m0 ** (m / 2) * integral / sn.C)
    return dict(D=D, life_s=T / D if D > 0 else np.inf, rms=float(np.sqrt(m0)),
                EP=float(EP), params=p)


def steinberg_damage(sigma_rms, fn, T, sn):
    """Steinberg three-band damage for a single resonance.

    1 sigma for 68.3 %, 2 sigma for 27.1 %, 3 sigma for 4.33 % of the time.
    """
    return float(sum(fn * T * frac / float(sn.cycles(k * sigma_rms))
                     for k, frac in ((1, 0.683), (2, 0.271), (3, 0.0433))))


def equivalent_test_time(T_service, W_service, W_test, m):
    """Test duration giving the same damage as service, for ASD levels W (g^2/Hz).

    T_test = T_service * (W_service / W_test) ** (m / 2), with m the S-N slope.
    Valid for a single-resonance response with the same spectral shape.
    """
    return T_service * (W_service / W_test) ** (m / 2.0)


# --------------------------------------------------------------------------
# Fatigue damage spectrum (FDS)
# --------------------------------------------------------------------------
def _fds_metric(z, wn, metric):
    if metric == "pseudo_velocity":
        return z * wn
    if metric == "rel_disp":
        return z
    raise ValueError("metric must be 'pseudo_velocity' or 'rel_disp'")


def fds_random(f, asd_g, T, fns, Q=10.0, b=8.0, metric="pseudo_velocity", g=9.81):
    """Fatigue damage spectrum of a random base acceleration ASD (g^2/Hz).

    For each natural frequency fn, an SDOF oscillator with amplification Q is
    driven by the ASD. Its response (pseudo velocity omega_n*z, proportional to
    modal stress by the stress-velocity relation, or relative displacement z) is
    narrow-band Gaussian, so the damage over T seconds is the Rayleigh result with
    S-N slope b and constant 1:  D = nu0 * T * (sqrt(2) s_rms)^b * Gamma(1 + b/2).

    The absolute values are relative (C = 1, unit stress per unit metric). Compare
    FDS curves of two environments with the same Q, b and metric: the ratio is the
    damage ratio at each frequency. Returns an array of D(fn).
    """
    from .sdof import rel_disp_frf
    f = np.asarray(f, float)
    Ga = np.asarray(asd_g, float) * g ** 2
    out = []
    zeta = 1.0 / (2.0 * Q)
    for fn in np.atleast_1d(fns):
        wn = 2 * np.pi * fn
        Gz = np.abs(rel_disp_frf(f, fn, zeta)) ** 2 * Ga
        Gs = _fds_metric(1.0, wn, metric) ** 2 * Gz
        m0 = trapezoid(Gs, f)
        m2 = trapezoid(f ** 2 * Gs, f)
        nu0 = np.sqrt(m2 / m0) if m0 > 0 else fn
        out.append(nu0 * T * (np.sqrt(2.0 * m0)) ** b * G_(1 + b / 2.0))
    return np.asarray(out)


def fds_time(accel, fs, fns, Q=10.0, b=8.0, metric="pseudo_velocity"):
    """Fatigue damage spectrum of a base acceleration time history (m/s^2).

    Rainflow-counts the SDOF response at each fn and sums Miner damage with
    S-N slope b and constant 1 (amplitudes of the chosen metric). Comparable with
    fds_random for the same Q, b and metric (accel in m/s^2 there means g=9.81).
    """
    from .sdof import sdof_response
    sn = SN(1.0, b)
    out = []
    for fn in np.atleast_1d(fns):
        z = sdof_response(accel, fs, fn, Q, "rel_disp")
        out.append(rainflow_damage(_fds_metric(z, 2 * np.pi * fn, metric), sn))
    return np.asarray(out)


# --------------------------------------------------------------------------
# Time history: rainflow
# --------------------------------------------------------------------------
def turning_points(x):
    """Reduce a signal to its peaks and valleys (plus end points)."""
    x = np.asarray(x, float)
    d = np.diff(x)
    keep = np.concatenate([[True], d[1:] * d[:-1] < 0, [True]])
    tp = x[keep]
    # drop repeated values created by plateaus
    return tp[np.concatenate([[True], np.diff(tp) != 0])]


def rainflow(x):
    """ASTM E1049 rainflow count. Returns arrays (ranges, means, counts)."""
    stack, cyc = [], []
    for p in turning_points(x):
        stack.append(p)
        while len(stack) >= 3:
            X = abs(stack[-1] - stack[-2])
            Y = abs(stack[-2] - stack[-3])
            if X < Y:
                break
            if len(stack) == 3:
                cyc.append((Y, 0.5 * (stack[-2] + stack[-3]), 0.5))
                stack.pop(0)
            else:
                cyc.append((Y, 0.5 * (stack[-2] + stack[-3]), 1.0))
                del stack[-3:-1]
    for a, b in zip(stack[:-1], stack[1:]):
        cyc.append((abs(b - a), 0.5 * (a + b), 0.5))
    if not cyc:
        return np.array([]), np.array([]), np.array([])
    r, mn, c = (np.array(v) for v in zip(*cyc))
    return r, mn, c


def rainflow_damage(x, sn, Su=None):
    """Miner damage of a stress time history. Goodman correction if Su is given."""
    r, mn, c = rainflow(x)
    if r.size == 0:
        return 0.0
    sa = r / 2.0
    if Su is not None:
        sa = goodman(sa, mn, Su)
    return float(np.sum(c / sn.cycles(sa)))
