"""End-to-end demo on synthetic data: test -> correlation -> fatigue.

A three-mode "specimen" on a shaker is the TEST truth. A slightly wrong "FEA"
model (frequencies 2-4 % off, perturbed mode shapes) is correlated against it:

1. random-vibration test data -> PSD, transmissibility (H1), coherence
2. modal parameters: half-power vs curve fit vs truth
3. modal table: frequency error and MAC
4. response PSD and Grms: test vs FEA
5. shock: half-sine base pulse -> SRS / pseudo velocity, test vs FEA
6. stress PSD -> Dirlik / narrow-band / Steinberg / rainflow damage

Run from the package folder:  python examples/demo_correlation.py
Replace the synthetic arrays with your lab data and ANSYS exports to reuse it.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vibtools import SN, fatigue, handcalc, modal, sdof, shock, spectral  # noqa: E402

OUT = os.path.join(os.environ.get("VIBTOOLS_OUT", os.path.dirname(os.path.abspath(__file__))), "demo_output.png")
G0 = 9.81

# ---------------------------------------------------------------- models
truth = dict(fn=np.array([85.0, 240.0, 510.0]), zeta=np.array([0.020, 0.030, 0.040]),
             gamma=np.array([0.80, 0.35, 0.30]),
             phi=np.array([[1.00, -0.70, 0.40],      # sensor A (tip)
                           [0.45, 0.50, -0.80],      # sensor B (mid)
                           [0.20, 0.90, 0.70]]))     # sensor C (base side)
rng = np.random.default_rng(4)
fea = dict(fn=np.array([88.0, 232.0, 522.0]), zeta=truth["zeta"], gamma=np.array([0.78, 0.37, 0.28]),
           phi=truth["phi"] * (1 + 0.04 * rng.standard_normal((3, 3))))
S_shape = np.array([[1.0, 0.55, 0.20]])             # stress shape; scaled below

# ------------------------------------------------- 1. random test on the shaker
fs, T = 8192.0, 120.0
profile = [(20, 0.01), (80, 0.04), (500, 0.04), (2000, 0.01)]     # g^2/Hz
f_in = np.linspace(0, fs / 2, 8000)
asd_in = spectral.asd_from_breakpoints(profile, f_in)
t, base = spectral.synth_gaussian(f_in, asd_in, T, fs, seed=11)             # g
resp = sdof.modal_response(base, fs, truth["fn"], truth["zeta"], truth["gamma"],
                           truth["phi"], "abs_acc")
resp += 0.05 * rng.standard_normal(resp.shape)                              # sensor noise
print(f"Input Grms {spectral.grms(*spectral.psd_welch(base, fs), 20, 2000):.2f} g "
      f"(profile {spectral.grms_from_breakpoints(profile):.2f} g)")

f, H1, H2, coh = spectral.frf_estimates(base, resp[0], fs, df=0.5)       # sensor A
H_all = [spectral.frf_estimates(base, r, fs, df=0.5)[1] for r in resp]   # all sensors
_, Pt = spectral.psd_welch(resp[0], fs, df=1.0)
fp, _ = spectral.psd_welch(resp[0], fs, df=1.0)

# ------------------------------------------------------ 2. modal parameters
# Damping is read from the sensor that responds most strongly in each mode's window,
# because a weak peak sitting on the skirt of its neighbours biases a single-mode fit.
windows = [(65, 110), (200, 285), (440, 600)]
print("\nMode   truth fn/zeta     half-power fn/zeta     curve-fit fn/zeta     sensor")
f_test, z_test = [], []
for k, w in enumerate(windows):
    m_w = (f >= w[0]) & (f <= w[1])
    s_best = int(np.argmax([np.abs(H[m_w]).max() for H in H_all]))
    Hk = H_all[s_best]
    hp = modal.half_power_damping(f, np.abs(Hk), w)
    ft = modal.fit_transmissibility(f, np.abs(Hk), hp["fn"], hp["zeta"], w)
    f_test.append(hp["fn"]); z_test.append(hp["zeta"])   # half-power values used below
    print(f"{k + 1:>4}   {truth['fn'][k]:6.1f}/{truth['zeta'][k]:.3f}     "
          f"{hp['fn']:6.1f}/{hp['zeta']:.3f}         {ft['fn']:6.1f}/{ft['zeta']:.3f}"
          f"         {'ABC'[s_best]}")

# ----------------------------------------------------- 3. modal correlation
phi_test = truth["phi"] * (1 + 0.03 * rng.standard_normal((3, 3)))   # "identified" shapes
rows = modal.correlation_table(np.array(f_test), fea["fn"], phi_test, fea["phi"])
print("\nModal correlation (test vs FEA)")
print(modal.format_table(rows))

# ------------------------------------------- 4. response PSD / Grms, test vs FEA
T_fea = sdof.modal_transmissibility(fp, fea["fn"], z_test, fea["gamma"], fea["phi"])[0]
asd_f = spectral.asd_from_breakpoints(profile, fp)
P_fea = np.abs(T_fea) ** 2 * asd_f
met = modal.psd_metrics(fp, Pt, P_fea, 20, 2000)
print(f"\nSensor A response: test {met['grms_test']:.2f} g, FEA {met['grms_fea']:.2f} g "
      f"({met['grms_diff_pct']:+.1f} %); peak at {met['peak_f_test']:.0f} Hz (test) vs "
      f"{met['peak_f_fea']:.0f} Hz (FEA)")

# --------------------------------------------------------------- 5. shock
fs_s = 20000.0
_, pulse = shock.half_sine(30.0, 0.006, fs_s, lead=0.005, tail=0.05)
fr = shock.octave_freqs(20, 1000, 12)
y_test = sdof.modal_response(pulse, fs_s, truth["fn"], truth["zeta"], truth["gamma"],
                             truth["phi"], "abs_acc")[0]
y_fea = sdof.modal_response(pulse, fs_s, fea["fn"], z_test, fea["gamma"], fea["phi"], "abs_acc")[0]
s_test = shock.srs(y_test * G0, fs_s, fr, 10)
s_fea = shock.srs(y_fea * G0, fs_s, fr, 10)
db, ok = shock.srs_compare(s_fea.acc_max, s_test.acc_max, tol_db=3.0)
print(f"\nShock SRS at sensor A (Q=10): {ok.mean() * 100:.0f} % of 1/12-octave points within +/-3 dB; "
      f"worst {np.nanmax(np.abs(db)):.1f} dB")
print(f"Peak response: test {np.abs(y_test).max():.1f} g, FEA {np.abs(y_fea).max():.1f} g")

# --------------------------------------------------------- 6. fatigue (random)
fg = np.linspace(5, 1500, 30000)
asd_ms2 = spectral.asd_from_breakpoints(profile, fg) * G0**2
H_true = sdof.modal_rel_frf(fg, truth["fn"], truth["zeta"], truth["gamma"], S_shape)[0]
G_true = np.abs(H_true) ** 2 * asd_ms2
scale = 25.0 / np.sqrt(trapezoid(G_true, fg))                 # calibrate: 25 MPa rms (illustrative)
S_true = S_shape * scale
G_true *= scale**2
H_fea = sdof.modal_rel_frf(fg, fea["fn"], z_test, fea["gamma"], S_shape * scale)[0]
G_fea = np.abs(H_fea) ** 2 * asd_ms2

sn = SN.from_points((200.0, 1e5), (100.0, 1e7))                 # illustrative S-N (MPa)
Ttest = 2 * 3600.0
res = {}
for name, G in (("test (truth)", G_true), ("FEA", G_fea)):
    mom = fatigue.spectral_moments(fg, G)
    d_dk = fatigue.dirlik_damage(fg, G, Ttest, sn)
    d_nb = fatigue.narrowband_damage(fg, G, Ttest, sn)
    fn_dom = fg[np.argmax(G)]
    d_st = fatigue.steinberg_damage(np.sqrt(mom[0]), fn_dom, Ttest, sn)
    res[name] = (np.sqrt(mom[0]), d_dk["D"], d_nb, d_st)
_, x_mc = spectral.synth_gaussian(fg, G_true, 600.0, 2048.0, seed=5)
d_mc = fatigue.rainflow_damage(x_mc, sn) * Ttest / 600.0
print("\nStress fatigue, 2 h at this level (S-N illustrative)")
print(f"{'':14}{'rms MPa':>9}{'Dirlik':>11}{'Narrow':>11}{'Steinberg':>11}")
for k, v in res.items():
    print(f"{k:14}{v[0]:9.1f}{v[1]:11.2e}{v[2]:11.2e}{v[3]:11.2e}")
print(f"{'rainflow MC':14}{'':9}{d_mc:11.2e}   (Monte-Carlo check on the test PSD)")
ratio = res["FEA"][1] / res["test (truth)"][1]
print(f"FEA/test damage ratio {ratio:.2f}  -> damage scales as stress^{sn.m:.1f}, "
      f"so report a range, not a single life")

# hand check
hc = handcalc.quick_check("Al 6061-T6", 2.0, accel_peak_g=float(np.abs(resp[0]).max()),
                          fn=truth["fn"][0])
print(f"\nStress-velocity hand check on sensor A peak ({np.abs(resp[0]).max():.0f} g at "
      f"{truth['fn'][0]:.0f} Hz, K=2, Al): {hc['sigma'] / 1e6:.0f} MPa order of magnitude")

# ------------------------------------------------------------------- plots
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
a = ax[0, 0]
a.semilogy(f, np.abs(H1), lw=0.8, label="test H1")
a.semilogy(fp, np.abs(T_fea), lw=1.2, label="FEA")
a.set(xlim=(20, 700), ylim=(0.1, 100), xlabel="Hz", ylabel="|T| sensor A / base",
      title="Transmissibility")
a2 = a.twinx(); a2.plot(f, coh, color="0.7", lw=0.6); a2.set(ylim=(0, 1.05), ylabel="coherence")
a.legend(loc="upper right")
a = ax[0, 1]
a.loglog(fp, Pt, lw=0.8, label=f"test ({met['grms_test']:.1f} g rms)")
a.loglog(fp, P_fea, lw=1.2, label=f"FEA ({met['grms_fea']:.1f} g rms)")
a.loglog(fp, asd_f, "k--", lw=0.8, label="input")
a.set(xlim=(20, 2000), ylim=(1e-4, 10), xlabel="Hz", ylabel="g$^2$/Hz", title="Response PSD, sensor A")
a.legend()
a = ax[1, 0]
a.loglog(fr, s_test.pseudo_vel / 0.0254, label="test")
a.loglog(fr, s_fea.pseudo_vel / 0.0254, "--", label="FEA")
a.set(xlabel="Hz", ylabel="pseudo velocity [in/s]", title="Shock SRS (Q=10), sensor A")
a.legend()
a = ax[1, 1]
a.loglog(fg, G_true, label="test (truth)")
a.loglog(fg, G_fea, "--", label="FEA")
a.set(xlim=(20, 1000), ylim=(1e-4, 1e3), xlabel="Hz", ylabel="MPa$^2$/Hz",
      title=f"Hotspot stress PSD (Dirlik FEA/test = {ratio:.2f})")
a.legend()
from matplotlib.ticker import NullFormatter
for a in ax.ravel():
    a.xaxis.set_minor_formatter(NullFormatter())
fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"\nSaved {OUT}")
