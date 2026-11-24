"""Reference solution for an FEA benchmark: cantilever with tip mass on a shaker.

Aluminium 6061-T6 strip, clamped at the root, 50 g point mass at the free end, base
excitation in the thickness direction. The reference is a converged Euler-Bernoulli
beam finite-element model (400 Hermite elements) solved by mode superposition with
the same damping you will use in ANSYS. Run:

    python examples/benchmark_cantilever.py            # prints the reference table
    python examples/benchmark_cantilever.py --plot     # also writes benchmark_cantilever.png

Build the same model in your FEA code (ANSYS, CalculiX, ...) and compare with the printed values.
"""
import os
import sys

import numpy as np
import scipy.linalg as sl
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vibtools import SN, fatigue, sdof, shock  # noqa: E402

# ------------------------------------------------------------------ inputs (SI)
E, RHO, NU = 68.9e9, 2700.0, 0.33          # Al 6061-T6
L, B, H = 0.200, 0.025, 0.005              # length, width, thickness
M_TIP = 0.050                              # tip point mass, no rotary inertia
ZETA = 0.02                                # all modes
G = 9.81
NEL = 400
X_GAUGE = 0.010                            # m from the root, for stress comparison
PSD_LEVEL, F1, F2 = 0.04, 20.0, 2000.0     # flat base ASD, g^2/Hz
T_RANDOM = 3600.0                          # s, for damage
SN_CURVE = SN.from_points((250.0, 1e4), (100.0, 1e7))   # ILLUSTRATIVE amplitude S-N (MPa)
PULSE_G, PULSE_MS = 30.0, 11.0             # half-sine shock


def beam_matrices(EI, mA, Le, n):
    k = EI / Le**3 * np.array([[12, 6 * Le, -12, 6 * Le], [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
                               [-12, -6 * Le, 12, -6 * Le], [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2]])
    m = mA * Le / 420 * np.array([[156, 22 * Le, 54, -13 * Le], [22 * Le, 4 * Le**2, 13 * Le, -3 * Le**2],
                                  [54, 13 * Le, 156, -22 * Le], [-13 * Le, -3 * Le**2, -22 * Le, 4 * Le**2]])
    ndof = 2 * (n + 1)
    K = np.zeros((ndof, ndof)); M = np.zeros((ndof, ndof))
    for e in range(n):
        s = slice(2 * e, 2 * e + 4)
        K[s, s] += k; M[s, s] += m
    return K, M


def solve(m_tip=M_TIP, I_axis="thin"):
    I = B * H**3 / 12 if I_axis == "thin" else H * B**3 / 12
    EI, mA, Le = E * I, RHO * B * H, L / NEL
    K, M = beam_matrices(EI, mA, Le, NEL)
    M[-2, -2] += m_tip                        # tip translation
    K, M = K[2:, 2:], M[2:, 2:]               # clamp root (w = w' = 0)
    w2, P = sl.eigh(K, M, subset_by_index=[0, 9])
    P = P / np.sqrt(np.einsum("ij,jk,ki->i", P.T, M, P))
    r = np.zeros(M.shape[0]); r[0::2] = 1.0   # rigid translation of every node
    gam = P.T @ M @ r
    # root bending stress per unit modal coordinate: sigma = E*(h/2)*w''(0)
    # w'' at x=0 of element 1 from Hermite shape functions (w0 = th0 = 0 at the clamp)
    d2N = np.array([-6 / Le**2, -4 / Le, 6 / Le**2, -2 / Le])
    c = (H if I_axis == "thin" else B) / 2
    phi_root_curv = np.array([d2N[2] * P[0, i] + d2N[3] * P[1, i] for i in range(P.shape[1])])
    sig_shape = E * c * phi_root_curv / 1e6   # MPa per unit q
    # surface bending stress 10 mm from the root (free of the clamp singularity in a solid model)
    n10 = int(round(X_GAUGE / Le))           # node index along the beam (root = node 0)
    i0 = 2 * n10 - 2                          # reduced DOF index of w at that node
    d2N0 = np.array([-6 / Le**2, -4 / Le, 6 / Le**2, -2 / Le])
    sig10 = np.array([E * c * (d2N0 @ P[i0:i0 + 4, i]) / 1e6 for i in range(P.shape[1])])
    tip = P[-2, :]
    total_mass = RHO * B * H * L + m_tip
    return dict(f=np.sqrt(w2) / 2 / np.pi, gam=gam, meff=gam**2, tip=tip, sig=sig_shape, sig10=sig10,
                total_mass=total_mass, M=M, P=P)


def main(plot=False):
    s = solve()
    f, gam, tip, sig, sig10 = s["f"], s["gam"], s["tip"], s["sig"], s["sig10"]
    nm = 6
    print("=" * 72)
    print("Benchmark: Al 6061-T6 cantilever 200 x 25 x 5 mm, 50 g tip mass, zeta = 2 %")
    print("=" * 72)
    # closed-form check without tip mass
    s0 = solve(m_tip=0.0)
    lam = 1.87510407
    f_cf = lam**2 / (2 * np.pi * L**2) * np.sqrt(E * B * H**3 / 12 / (RHO * B * H))
    print(f"check, no tip mass: f1 = {s0['f'][0]:.2f} Hz vs closed form {f_cf:.2f} Hz")
    lat = solve(I_axis="strong")
    print(f"in-plane (width-direction) bending f1 = {lat['f'][0]:.1f} Hz (not excited by thickness-axis input)")
    beta = 0.291                               # torsion constant factor for b/h = 5
    Gm = E / (2 * (1 + NU)); J = beta * B * H**3; Ip = B * H * (B**2 + H**2) / 12
    print(f"first torsion mode (uniform bar, point mass adds no inertia) = "
          f"{np.sqrt(Gm * J / (RHO * Ip)) / (4 * L):.0f} Hz")
    print("\nThickness-direction bending modes")
    print(" mode   f (Hz)    m_eff (g)   cum. %")
    cum = np.cumsum(s["meff"]) / s["total_mass"] * 100
    for i in range(4):
        print(f"  {i+1:2d}  {f[i]:8.2f}   {s['meff'][i]*1e3:8.2f}   {cum[i]:6.1f}")
    print(f" total moving mass {s['total_mass']*1e3:.1f} g")

    # harmonic: 1 g base
    fr = np.linspace(5, 2000, 40000)
    W = 2 * np.pi * fr
    q = np.array([-gam[i] * G / (4 * np.pi**2 * f[i]**2 - W**2 + 2j * ZETA * 2 * np.pi * f[i] * W)
                  for i in range(nm)])
    z_tip = (tip[:nm, None] * q).sum(0)
    T_tip = np.abs(1 - W**2 * z_tip / G)
    sig_root = np.abs((sig[:nm, None] * q).sum(0))
    sig_10 = np.abs((sig10[:nm, None] * q).sum(0))
    i1 = np.argmax(T_tip * (fr < 1.5 * f[0]))
    print("\nHarmonic, 1 g base, MSUP 6 modes")
    print(f"  tip transmissibility peak  {T_tip[i1]:.2f} at {fr[i1]:.2f} Hz")
    print(f"  root stress amplitude      {sig_root[i1]:.2f} MPa per g at {fr[i1]:.2f} Hz")
    print(f"  root stress per g at 10 Hz (quasi-static) {sig_root[np.argmin(abs(fr-10))]:.3f} MPa")
    print(f"  stress at x = 10 mm: {sig_10[i1]:.2f} MPa per g at the peak, "
          f"{sig_10[np.argmin(abs(fr-10))]:.3f} MPa per g at 10 Hz")

    # random
    asd = np.where((fr >= F1) & (fr <= F2), PSD_LEVEL, 0.0)
    G_tip = T_tip**2 * asd
    G_sig = sig_root**2 * asd                 # MPa^2/Hz (|sigma per g|^2 * g^2/Hz)
    tip_grms = np.sqrt(trapezoid(G_tip, fr)); sig_rms = np.sqrt(trapezoid(G_sig, fr))
    miles = sdof.miles_grms(f[0], 1 / (2 * ZETA), PSD_LEVEL)
    gam_tip = gam[0] * tip[0]
    D = fatigue.dirlik_damage(fr, G_sig, T_RANDOM, SN_CURVE)
    print(f"\nRandom, flat {PSD_LEVEL} g^2/Hz {F1:.0f}-{F2:.0f} Hz ({np.sqrt(PSD_LEVEL*(F2-F1)):.2f} Grms input)")
    print(f"  tip absolute Grms (1 sigma)  {tip_grms:.2f} g")
    print(f"  Miles check, mode 1 only:    {miles:.2f} g x participation at tip {abs(gam_tip):.3f} = "
          f"{miles*abs(gam_tip):.2f} g")
    print(f"  root stress RMS (1 sigma)    {sig_rms:.2f} MPa")
    print(f"  stress RMS at x = 10 mm      {np.sqrt(trapezoid(sig_10**2 * asd, fr)):.2f} MPa")
    print(f"  Dirlik damage, 1 h, illustrative S-N: {D['D']:.3e}  (life {D['life_s']/3600:.3g} h)")

    # stress-velocity check at the mode-1 peak
    v_rel_tip = (W[i1] * np.abs(z_tip[i1]))
    rho_c = np.sqrt(E * RHO) / 1e6
    print("\nStress-velocity check at the mode-1 peak, 1 g input")
    print(f"  tip relative velocity {v_rel_tip*1e3:.1f} mm/s,  rho*c = {rho_c:.2f} MPa s/m")
    print(f"  effective K = sigma_root / (rho c v_tip) = {sig_root[i1] / (rho_c * v_rel_tip):.2f} "
          "(uniform cantilever without tip mass: about sqrt(3) = 1.73)")

    # shock
    fs = 50000.0
    t, a = shock.half_sine(PULSE_G * G, PULSE_MS / 1000, fs, tail=0.15)
    zeta6 = np.full(nm, ZETA)
    out = sdof.modal_response(a, fs, f[:nm], zeta6, gam[:nm], np.vstack([tip[:nm], sig[:nm], sig10[:nm]]),
                              "rel_disp")
    acc_tip = sdof.modal_response(a, fs, f[:nm], zeta6, gam[:nm], tip[None, :nm], "abs_acc")[0]
    print(f"\nShock, half-sine {PULSE_G:.0f} g {PULSE_MS:.0f} ms, modal transient 6 modes")
    print(f"  tip peak absolute acceleration {np.abs(acc_tip).max()/G:.1f} g")
    print(f"  root peak stress               {np.abs(out[1]).max():.1f} MPa")
    print(f"  peak stress at x = 10 mm       {np.abs(out[2]).max():.1f} MPa")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8))
        ax[0].semilogy(fr, T_tip, color="#2a78d6", lw=2)
        ax[0].set(xlim=(5, 600), xlabel="Hz", ylabel="|T| tip / base", title="Tip transmissibility (1 g base)")
        ax[1].loglog(fr[asd > 0], G_sig[asd > 0], color="#eb6834", lw=2)
        ax[1].set(xlabel="Hz", ylabel="MPa²/Hz", title=f"Root stress PSD ({sig_rms:.1f} MPa rms)")
        for a_ in ax:
            a_.grid(True, color="#e4e3df"); a_.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        png = os.path.join(os.environ.get("VIBTOOLS_OUT", os.path.dirname(os.path.abspath(__file__))),
                           "benchmark_cantilever.png")
        fig.savefig(png, dpi=200)
        print(f"\nwrote {png}")


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)
