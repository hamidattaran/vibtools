"""Stress-velocity hand checks (Hunt, Gaberson, Bateman; see Irvine, rev. Q).

sigma_max = K * rho*c * v     with rho*c = sqrt(E * rho)

Use for screening and for sanity-checking FEA, not for certification or local
stress concentrations. Any consistent unit system works; SI helpers below.
"""
from __future__ import annotations

import numpy as np

# Typical handbook values (SI): E [Pa], rho [kg/m^3], yield [Pa]. Illustrative only;
# use your own material database for real work (polymers vary strongly with
# temperature, rate and fibre orientation, so none are listed).
MATERIALS = {
    "Al 6061-T6": dict(E=68.9e9, rho=2700.0, yield_=276e6),
    "Al 7075-T6": dict(E=71.7e9, rho=2810.0, yield_=503e6),
    "Steel A36": dict(E=200e9, rho=7850.0, yield_=250e6),
    "Stainless 304": dict(E=193e9, rho=8000.0, yield_=215e6),
    "Ti-6Al-4V": dict(E=114e9, rho=4430.0, yield_=880e6),
}

# Representative K values (Irvine rev. Q, Hunt, Bateman, Sweitzer et al.).
K_FACTORS = {
    "rod, longitudinal": 1.0,
    "beam, solid rectangular (cantilever / simply supported)": np.sqrt(3.0),
    "beam, solid circular (cantilever / simply supported)": 2.0,
    "beams and plates, homogeneous (general)": 2.0,
    "complex equipment, incl. stress concentration (low)": 4.0,
    "complex equipment, incl. stress concentration (high)": 8.0,
}


def rho_c(E, rho):
    """Characteristic impedance sqrt(E*rho). SI: Pa*s/m."""
    return float(np.sqrt(E * rho))


def pseudo_velocity(rel_disp_max, fn):
    """Pseudo velocity = 2*pi*fn * peak relative displacement."""
    return 2.0 * np.pi * fn * rel_disp_max


def velocity_from_accel(accel_peak, fn):
    """Approximate pseudo velocity from peak absolute acceleration at resonance."""
    return accel_peak / (2.0 * np.pi * fn)


def stress_from_velocity(K, rho_c_value, v, fn=None, f_excite=None):
    """Peak modal stress K*rho*c*v. Pass fn and f_excite for off-resonance scaling."""
    scale = 1.0 if fn is None or f_excite is None else fn / f_excite
    return scale * K * rho_c_value * v


def velocity_limit(sigma_allow, K, rho_c_value):
    """Modal velocity at which the allowable stress is reached."""
    return sigma_allow / (K * rho_c_value)


def drop_velocity(h, g=9.81):
    """Impact velocity from drop height (SI by default)."""
    return float(np.sqrt(2.0 * g * h))


def drop_sdof(h, fn, g=9.81):
    """SDOF initial-velocity model: velocity, peak rel. displacement, peak accel."""
    v = drop_velocity(h, g)
    w = 2.0 * np.pi * fn
    return dict(v=v, disp=v / w, accel=w * v, accel_g=w * v / g)


def quick_check(material, K, accel_peak_g, fn, sigma_fea=None):
    """Hand estimate of peak stress from a resonant acceleration (SI).

    Returns dict(v, sigma, ratio_fea_to_hand when sigma_fea is given).
    """
    mat = MATERIALS[material]
    v = velocity_from_accel(accel_peak_g * 9.81, fn)
    sigma = stress_from_velocity(K, rho_c(mat["E"], mat["rho"]), v)
    out = dict(v=v, sigma=sigma, yield_=mat["yield_"], sigma_over_yield=sigma / mat["yield_"])
    if sigma_fea is not None:
        out["ratio_fea_to_hand"] = sigma_fea / sigma
    return out
