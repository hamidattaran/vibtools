"""Single-DOF and modal base-excitation helpers.

Conventions
-----------
* Base excitation: the base acceleration is the input.
* ``abs_acc``  : absolute acceleration of the SDOF mass (same units as the input).
* ``rel_disp`` : displacement of the mass relative to the base, from
  ``z'' + 2*zeta*wn*z' + wn^2*z = -a_base``. If the input is in m/s^2 the result
  is in metres.
* Q = 1 / (2 * zeta).

Time-domain responses use a ramp-invariant (first-order-hold) digital filter,
the same family as the Smallwood algorithm used for shock response spectra.
It is accurate for frequencies up to roughly fs/8.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import cont2discrete, lfilter


def ramp_invariant_coeffs(fn, Q, fs, kind="abs_acc"):
    """Digital filter coefficients (b, a) for an SDOF base-excitation response."""
    w = 2.0 * np.pi * fn
    z = 1.0 / (2.0 * Q)
    den = [1.0, 2.0 * z * w, w * w]
    if kind == "abs_acc":
        num = [2.0 * z * w, w * w]
    elif kind == "rel_disp":
        num = [-1.0]
    else:
        raise ValueError("kind must be 'abs_acc' or 'rel_disp'")
    nd, dd, _ = cont2discrete((num, den), 1.0 / fs, method="foh")
    return np.atleast_1d(nd).ravel(), np.atleast_1d(dd).ravel()


def sdof_response(base_acc, fs, fn, Q, kind="abs_acc"):
    """Time-domain SDOF response to a base acceleration time history."""
    b, a = ramp_invariant_coeffs(fn, Q, fs, kind)
    return lfilter(b, a, np.asarray(base_acc, float))


def transmissibility(f, fn, zeta):
    """Complex absolute-acceleration transmissibility of an SDOF system."""
    r = np.asarray(f, float) / fn
    return (1.0 + 2j * zeta * r) / (1.0 - r**2 + 2j * zeta * r)


def rel_disp_frf(f, fn, zeta):
    """Complex relative displacement per unit base acceleration (m per m/s^2)."""
    w = 2.0 * np.pi * np.asarray(f, float)
    wn = 2.0 * np.pi * fn
    return -1.0 / (wn**2 - w**2 + 2j * zeta * wn * w)


def response_psd(f, asd, fn, zeta):
    """Absolute-acceleration response PSD of an SDOF system to a base ASD."""
    return np.abs(transmissibility(f, fn, zeta)) ** 2 * np.asarray(asd, float)


def miles_grms(fn, Q, asd_at_fn):
    """Miles' equation: response Grms of an SDOF system on a flat-ish ASD.

    ``asd_at_fn`` in g^2/Hz gives Grms in g. Accurate for Q >~ 10 and a PSD that
    is smooth around fn.
    """
    return float(np.sqrt(np.pi / 2.0 * fn * Q * asd_at_fn))


# --------------------------------------------------------------------------
# Modal superposition for base excitation
# --------------------------------------------------------------------------
def participation_factors(phi, M, r=None):
    """Modal participation factors for mass-normalised mode shapes.

    phi : (ndof, nmodes) mode shapes, normalised so that phi.T @ M @ phi = I.
    M   : (ndof, ndof) mass matrix.
    r   : influence vector for the base direction (default: ones).
    """
    phi = np.atleast_2d(phi)
    r = np.ones(phi.shape[0]) if r is None else np.asarray(r, float)
    return phi.T @ (np.asarray(M, float) @ r)


def modal_response(base_acc, fs, fn, zeta, gamma, shapes, quantity="rel_disp"):
    """Response of a base-excited MDOF system by modal superposition.

    Parameters
    ----------
    base_acc : base acceleration time history.
    fn, zeta, gamma : per-mode natural frequency (Hz), damping ratio and
        participation factor (arrays of length nmodes).
    shapes : (n_out, nmodes) matrix. For ``quantity='rel_disp'`` it maps modal
        coordinates to the output of interest (relative displacement, strain or
        stress shapes). For ``'abs_acc'`` it must be displacement mode shapes at
        the sensor DOFs.
    quantity : 'rel_disp' or 'abs_acc'.

    Returns an (n_out, nsamples) array.
    """
    base_acc = np.asarray(base_acc, float)
    fn = np.atleast_1d(np.asarray(fn, float))
    zeta = np.broadcast_to(np.asarray(zeta, float), fn.shape)
    gamma = np.broadcast_to(np.asarray(gamma, float), fn.shape)
    shapes = np.atleast_2d(np.asarray(shapes, float))
    out = np.zeros((shapes.shape[0], base_acc.size))
    for j in range(fn.size):
        Q = 1.0 / (2.0 * zeta[j])
        if quantity == "rel_disp":
            z = sdof_response(base_acc, fs, fn[j], Q, "rel_disp")
            out += np.outer(shapes[:, j] * gamma[j], z)
        elif quantity == "abs_acc":
            y = sdof_response(base_acc, fs, fn[j], Q, "abs_acc")
            out += np.outer(shapes[:, j] * gamma[j], y - base_acc)
        else:
            raise ValueError("quantity must be 'rel_disp' or 'abs_acc'")
    if quantity == "abs_acc":
        out += base_acc[None, :]
    return out


def modal_transmissibility(f, fn, zeta, gamma, shapes):
    """Complex absolute-acceleration transmissibility at each output DOF.

    T_i(f) = 1 + sum_j shapes[i, j] * gamma[j] * (T_j(f) - 1)
    Returns an (n_out, nf) complex array.
    """
    f = np.asarray(f, float)
    fn = np.atleast_1d(np.asarray(fn, float))
    zeta = np.broadcast_to(np.asarray(zeta, float), fn.shape)
    gamma = np.broadcast_to(np.asarray(gamma, float), fn.shape)
    shapes = np.atleast_2d(np.asarray(shapes, float))
    T = np.ones((shapes.shape[0], f.size), dtype=complex)
    for j in range(fn.size):
        Tj = transmissibility(f, fn[j], zeta[j]) - 1.0
        T += np.outer(shapes[:, j] * gamma[j], Tj)
    return T


def modal_rel_frf(f, fn, zeta, gamma, shapes):
    """Complex output per unit base acceleration (relative-displacement basis).

    Use with stress or strain shapes to get a stress FRF; |H|^2 * ASD gives the
    stress PSD that feeds the spectral fatigue methods.
    """
    f = np.asarray(f, float)
    fn = np.atleast_1d(np.asarray(fn, float))
    zeta = np.broadcast_to(np.asarray(zeta, float), fn.shape)
    gamma = np.broadcast_to(np.asarray(gamma, float), fn.shape)
    shapes = np.atleast_2d(np.asarray(shapes, float))
    H = np.zeros((shapes.shape[0], f.size), dtype=complex)
    for j in range(fn.size):
        H += np.outer(shapes[:, j] * gamma[j], rel_disp_frf(f, fn[j], zeta[j]))
    return H
