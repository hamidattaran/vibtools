"""Modal correlation helpers: MAC, frequency error, mode pairing, damping."""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import least_squares, linear_sum_assignment

from .sdof import transmissibility


def mac(phi_a, phi_b):
    """Modal Assurance Criterion matrix.

    phi_a: (ndof, na) and phi_b: (ndof, nb), both sampled at the SAME sensor
    DOFs and directions (reduce the FEA shapes to the test sensors first).
    Returns an (na, nb) matrix with entries in [0, 1].
    """
    A = np.atleast_2d(phi_a)
    B = np.atleast_2d(phi_b)
    num = np.abs(A.conj().T @ B) ** 2
    den = np.outer(np.real(np.sum(A.conj() * A, axis=0)),
                   np.real(np.sum(B.conj() * B, axis=0)))
    return num / den


def freq_error_pct(f_test, f_fea):
    """Percent error (f_fea - f_test) / f_test."""
    f_test = np.asarray(f_test, float)
    return 100.0 * (np.asarray(f_fea, float) - f_test) / f_test


def pair_modes(mac_matrix, min_mac=0.5):
    """Pair test modes (rows) with FEA modes (columns) by maximising total MAC.

    Returns a list of (i_test, j_fea, mac). Pairs below ``min_mac`` are dropped.
    """
    M = np.asarray(mac_matrix, float)
    r, c = linear_sum_assignment(-M)
    return [(int(i), int(j), float(M[i, j])) for i, j in zip(r, c) if M[i, j] >= min_mac]


def correlation_table(f_test, f_fea, phi_test, phi_fea, min_mac=0.5):
    """Frequency and MAC summary for paired modes (list of dicts)."""
    M = mac(phi_test, phi_fea)
    rows = []
    for i, j, m in pair_modes(M, min_mac):
        rows.append(dict(test_mode=i + 1, fea_mode=j + 1, f_test=float(f_test[i]),
                         f_fea=float(f_fea[j]),
                         err_pct=float(freq_error_pct(f_test[i], f_fea[j])), mac=m))
    return rows


def format_table(rows):
    """Plain-text table for the rows returned by ``correlation_table``."""
    head = f"{'Test':>4} {'FEA':>4} {'f_test':>9} {'f_FEA':>9} {'err %':>7} {'MAC':>6}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['test_mode']:>4} {r['fea_mode']:>4} {r['f_test']:>9.2f} "
                     f"{r['f_fea']:>9.2f} {r['err_pct']:>7.2f} {r['mac']:>6.3f}")
    return "\n".join(lines)


def half_power_damping(f, mag, f_range=None):
    """Natural frequency and damping from the half-power bandwidth of one peak.

    f_range = (lo, hi) limits the search to one resonance. ``mag`` is the FRF or
    transmissibility magnitude. Returns dict(fn, zeta, Q, f1, f2).
    Resolution matters: the peak must span several frequency lines.
    """
    f = np.asarray(f, float)
    mag = np.abs(np.asarray(mag))
    idx = np.arange(f.size)
    if f_range is not None:
        idx = idx[(f >= f_range[0]) & (f <= f_range[1])]
    ip = idx[np.argmax(mag[idx])]
    pk = mag[ip]
    level = pk / np.sqrt(2.0)

    i = ip
    while i > idx[0] and mag[i] > level:
        i -= 1
    j = ip
    while j < idx[-1] and mag[j] > level:
        j += 1
    if mag[i] > level or mag[j] > level:
        raise ValueError("half-power points fall outside f_range; widen it")
    f1 = f[i] + (level - mag[i]) * (f[i + 1] - f[i]) / (mag[i + 1] - mag[i])
    f2 = f[j - 1] + (mag[j - 1] - level) * (f[j] - f[j - 1]) / (mag[j - 1] - mag[j])

    fn = f[ip]
    if 0 < ip < f.size - 1:                       # parabolic peak refinement
        y0, y1, y2 = mag[ip - 1], mag[ip], mag[ip + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            fn = f[ip] + 0.5 * (y0 - y2) / den * (f[ip + 1] - f[ip])
    zeta = (f2 - f1) / (2.0 * fn)
    return dict(fn=float(fn), zeta=float(zeta), Q=float(1 / (2 * zeta)),
                f1=float(f1), f2=float(f2))


def fit_transmissibility(f, T, fn0, zeta0=0.02, f_range=None):
    """Fit fn, zeta and a scale to one resonance of a transmissibility magnitude.

    Fits |T| of an SDOF base-excitation model over ``f_range`` (default: fn0
    +/- 30%). More robust than half-power on noisy or coarse data. Returns
    dict(fn, zeta, Q, scale).
    """
    f = np.asarray(f, float)
    mag = np.abs(np.asarray(T))
    lo, hi = f_range if f_range is not None else (0.7 * fn0, 1.3 * fn0)
    m = (f >= lo) & (f <= hi)

    def resid(p):
        fn, z, s = p
        return s * np.abs(transmissibility(f[m], fn, z)) - mag[m]

    sol = least_squares(resid, [fn0, zeta0, 1.0],
                        bounds=([lo, 1e-4, 0.1], [hi, 0.5, 10.0]))
    fn, z, s = sol.x
    return dict(fn=float(fn), zeta=float(z), Q=float(1 / (2 * z)), scale=float(s))


def half_power_damping_psd(f, psd, f_range=None):
    """Half-power damping from a RESPONSE PSD peak (one mode, near-flat input).

    A PSD is proportional to |H|^2, so the half-power points sit at HALF the PSD
    peak (-3 dB), not at 0.707 of it. This takes sqrt(PSD) and applies the usual
    0.707 rule, which is the same thing. Valid when the input PSD is roughly flat
    across the peak; otherwise divide by the input PSD first (transmissibility^2).
    """
    return half_power_damping(f, np.sqrt(np.abs(np.asarray(psd, float))), f_range)


def log_decrement(x, fs, fn=None, band=None, skip_cycles=1, min_ratio=0.05, max_cycles=60):
    """Damping from a free decay (ring-down after an impact or a shock).

    The signal is band-pass filtered around one mode (``band`` = (lo, hi) Hz, or
    fn +/- 30 % when ``fn`` is given; with neither, fn is taken from the spectrum
    of the decay), positive peaks are found after the largest one, and ln(peak)
    is fitted against time. A straight line means viscous (linear) damping; if a
    straight line fits the peak amplitudes themselves better, the decay is
    friction-dominated (Coulomb) and one zeta does not describe it.

    Returns dict(fn, zeta, Q, delta, n_peaks, r2_log, r2_linear, t_peaks, a_peaks).
    """
    from scipy.signal import butter, find_peaks, sosfiltfilt

    x = np.asarray(x, float) - np.mean(x)
    if band is None:
        if fn is None:
            i0 = int(np.argmax(np.abs(x)))
            seg = x[i0:]
            spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
            ff = np.fft.rfftfreq(seg.size, 1 / fs)
            spec[ff < 1.0] = 0
            fn = float(ff[np.argmax(spec)])
        band = (0.7 * fn, 1.3 * fn)
    sos = butter(4, [band[0], band[1]], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    i0 = int(np.argmax(np.abs(y)))
    fguess = fn if fn is not None else 0.5 * (band[0] + band[1])
    pk, _ = find_peaks(y[i0:], distance=max(1, int(0.6 * fs / fguess)))
    pk = pk + i0
    if pk.size < 3 + skip_cycles:
        raise ValueError("too few decay cycles: check the band or record a longer ring-down")
    pk = pk[skip_cycles:]
    a = y[pk]
    keep = a > min_ratio * a[0]
    pk, a = pk[keep][:max_cycles], a[keep][:max_cycles]
    if pk.size < 3:
        raise ValueError("decay drops into the noise too quickly: lower min_ratio or improve the signal")
    # refine peak times by parabolic interpolation
    t = []
    for k in pk:
        y0, y1, y2 = y[k - 1], y[k], y[k + 1]
        den = y0 - 2 * y1 + y2
        t.append((k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)) / fs)
    t = np.asarray(t)
    la = np.log(a)
    s_log, c_log = np.polyfit(t, la, 1)
    s_lin, c_lin = np.polyfit(t, a, 1)
    r2 = lambda y_, yh: 1 - np.sum((y_ - yh) ** 2) / np.sum((y_ - y_.mean()) ** 2)
    fd = 1.0 / np.mean(np.diff(t))
    sigma = -s_log
    wd = 2 * np.pi * fd
    zeta = sigma / np.sqrt(sigma ** 2 + wd ** 2)
    return dict(fn=float(fd / np.sqrt(1 - zeta ** 2)), zeta=float(zeta), Q=float(1 / (2 * zeta)),
                delta=float(sigma / fd), n_peaks=int(pk.size),
                r2_log=float(r2(la, s_log * t + c_log)), r2_linear=float(r2(a, s_lin * t + c_lin)),
                t_peaks=t, a_peaks=a)


def fit_modes(f, H, fn0, zeta0=0.02, f_range=None, kind="transmissibility"):
    """Multi-mode curve fit of a complex FRF or transmissibility (close modes).

    Model (kind="transmissibility", base excitation, absolute acceleration):
        T(w) = c + sum_r R_r w^2 / (w_r^2 - w^2 + 2j zeta_r w_r w)
    Model (kind="frf", e.g. accelerance or receptance over a band):
        H(w) = c0 + c2 w^2 + sum_r R_r / (w_r^2 - w^2 + 2j zeta_r w_r w)
    R_r and the residual terms are complex and solved linearly at every step
    (variable projection); only fn_r and zeta_r are iterated. Give one starting
    frequency per mode in ``fn0``. Use the complex H1 (not its magnitude).

    Returns dict(fn, zeta, Q, residues, fit (model on f), rms_error_pct).
    """
    f = np.asarray(f, float)
    H = np.asarray(H, complex)
    fn0 = np.atleast_1d(np.asarray(fn0, float))
    lo, hi = f_range if f_range is not None else (0.8 * fn0.min(), 1.2 * fn0.max())
    m = (f >= lo) & (f <= hi)
    fm, Hm = f[m], H[m]
    w = 2 * np.pi * fm
    nm = fn0.size

    def basis(p, ww):
        fn, z = p[:nm], p[nm:]
        wr = 2 * np.pi * fn
        cols = []
        for r in range(nm):
            den = wr[r] ** 2 - ww ** 2 + 2j * z[r] * wr[r] * ww
            cols.append((ww ** 2 if kind == "transmissibility" else 1.0) / den)
        cols.append(np.ones_like(ww, dtype=complex))
        if kind != "transmissibility":
            cols.append(ww.astype(complex) ** 2)
        return np.column_stack(cols)

    def solve_lin(p, ww, Hw):
        A = basis(p, ww)
        coef, *_ = np.linalg.lstsq(A, Hw, rcond=None)
        return A, coef

    def resid(p):
        A, coef = solve_lin(p, w, Hm)
        e = (A @ coef - Hm) / (np.abs(Hm) + 1e-12 * np.abs(Hm).max())
        return np.concatenate([e.real, e.imag])

    p0 = np.concatenate([fn0, np.full(nm, zeta0)])
    lb = np.concatenate([np.full(nm, lo), np.full(nm, 1e-4)])
    ub = np.concatenate([np.full(nm, hi), np.full(nm, 0.5)])
    sol = least_squares(resid, p0, bounds=(lb, ub), x_scale=np.concatenate([fn0 * 0.01, np.full(nm, 0.01)]))
    A, coef = solve_lin(sol.x, w, Hm)
    fit_all = basis(sol.x, 2 * np.pi * f) @ coef
    err = np.sqrt(np.mean(np.abs(A @ coef - Hm) ** 2)) / np.sqrt(np.mean(np.abs(Hm) ** 2)) * 100
    fn, z = sol.x[:nm], sol.x[nm:]
    order = np.argsort(fn)
    return dict(fn=fn[order], zeta=z[order], Q=1 / (2 * z[order]), residues=coef[:nm][order],
                fit=fit_all, rms_error_pct=float(err))


def psd_metrics(f, psd_test, psd_fea, f1, f2):
    """Grms of each PSD over a band, their % difference, and peak-frequency shift."""
    f = np.asarray(f, float)
    m = (f >= f1) & (f <= f2)
    g_t = float(np.sqrt(trapezoid(np.asarray(psd_test)[m], f[m])))
    g_f = float(np.sqrt(trapezoid(np.asarray(psd_fea)[m], f[m])))
    f_t = f[m][np.argmax(np.asarray(psd_test)[m])]
    f_f = f[m][np.argmax(np.asarray(psd_fea)[m])]
    return dict(grms_test=g_t, grms_fea=g_f, grms_diff_pct=100 * (g_f - g_t) / g_t,
                peak_f_test=float(f_t), peak_f_fea=float(f_f))
