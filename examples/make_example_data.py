"""Synthetic example inputs for every vibtools command.

    python examples/make_example_data.py [output_folder]     # default: ./example_data
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vibtools import io, sdof, shock, spectral  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "example_data"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# 3-mode specimen (same as the demo)
fn = np.array([85.0, 240.0, 510.0]); zeta = np.array([0.02, 0.03, 0.04]); gam = np.array([0.8, 0.35, 0.3])
phi = np.array([[1.0, -0.7, 0.4], [0.45, 0.5, -0.8]])

# 1. shaker random run: time, control (g), tip (g), mid (g)
fs, T = 4096.0, 60.0
f = np.linspace(0, fs / 2, 8000)
asd = spectral.asd_from_breakpoints([(20, 0.01), (80, 0.04), (500, 0.04), (2000, 0.01)], f)
t, base = spectral.synth_gaussian(f, asd, T, fs, seed=11)
resp = sdof.modal_response(base, fs, fn, zeta, gam, phi, "abs_acc")
resp += 0.03 * rng.standard_normal(resp.shape)
np.savetxt(os.path.join(OUT, "shaker_random_run.csv"), np.column_stack([t, base, resp[0], resp[1]]),
           delimiter=",", header="Time [s],control [g],tip [g],mid [g]", comments="", fmt="%.9g")

# 2. shock: half-sine 30 g 11 ms on the table, response at the tip
fs2 = 20000.0
t2, a2 = shock.half_sine(30.0, 0.011, fs2, lead=0.01, tail=0.2)
tip2 = sdof.modal_response(a2 * 9.81, fs2, fn, zeta, gam, phi[:1], "abs_acc")[0] / 9.81
np.savetxt(os.path.join(OUT, "shock_halfsine.csv"), np.column_stack([t2, a2, tip2]),
           delimiter=",", header="Time [s],table [g],tip [g]", comments="", fmt="%.9g")

# 3. handle vibration, three axes in m/s^2: 75 Hz order + 225 Hz order + broadband
fs3, T3 = 5000.0, 20.0
t3 = np.arange(0, T3, 1 / fs3)
ax_ = 6 * np.sin(2 * np.pi * 75 * t3) + 2 * np.sin(2 * np.pi * 225 * t3 + 0.3) + rng.normal(0, 1.0, t3.size)
ay_ = 3 * np.sin(2 * np.pi * 75 * t3 + 1.0) + rng.normal(0, 0.8, t3.size)
az_ = 9 * np.sin(2 * np.pi * 75 * t3 + 0.5) + 4 * np.sin(2 * np.pi * 150 * t3) + rng.normal(0, 1.5, t3.size)
np.savetxt(os.path.join(OUT, "handle_3axis.csv"), np.column_stack([t3, ax_, ay_, az_]),
           delimiter=",", header="Time [s],ax [m/s2],ay [m/s2],az [m/s2]", comments="", fmt="%.9g")

# 4. stress PSD at a hotspot (two modes), Workbench-style tab-delimited text
fsp = np.linspace(20, 2000, 2000)
Hs = 1.2 * np.abs(sdof.rel_disp_frf(fsp, 85, 0.02)) / np.abs(sdof.rel_disp_frf(0.1, 85, 0.02)) \
    + 0.25 * np.abs(sdof.rel_disp_frf(fsp, 240, 0.03)) / np.abs(sdof.rel_disp_frf(0.1, 240, 0.03))
Gs = Hs ** 2 * spectral.asd_from_breakpoints([(20, 0.01), (80, 0.04), (500, 0.04), (2000, 0.01)], fsp)
with open(os.path.join(OUT, "stress_psd_hotspot.txt"), "w", encoding="utf-8") as fh:
    fh.write("Response PSD - Normal Stress\nFrequency [Hz]\tStress PSD [MPa²/Hz]\n")
    for a, b in zip(fsp, Gs):
        fh.write(f"{a:.4f}\t{b:.6e}\n")

# 5. mode tables: rows = modes; columns: freq, then DOF values at the test sensors
shapes_test = np.array([[1.00, 0.45, 0.20, -0.10], [-0.70, 0.50, 0.90, 0.30], [0.40, -0.80, 0.70, 0.55]])
shapes_fea = shapes_test * (1 + 0.05 * rng.standard_normal(shapes_test.shape))
for name, fr, sh in (("modes_test.csv", [84.9, 239.3, 513.6], shapes_test),
                     ("modes_fea.csv", [88.0, 232.0, 522.0], shapes_fea)):
    np.savetxt(os.path.join(OUT, name), np.column_stack([fr, sh]), delimiter=",",
               header="freq [Hz],s1,s2,s3,s4", comments="", fmt="%.5g")

# 6. UFF file with the measured transmissibility (H1 of the shaker run)
fq, H1, _, _ = spectral.frf_estimates(base, resp[0], fs, df=0.5)
io.write_uff58(os.path.join(OUT, "tip_transmissibility.uff"), fq, H1, func_type=5, response=(1, 3),
               y_label="Transmissibility", y_unit="g/g")

# 7. service ASD for an FDS comparison (lower level, longer)
fsv = np.linspace(10, 2000, 4000)
np.savetxt(os.path.join(OUT, "service_asd.csv"),
           np.column_stack([fsv, spectral.asd_from_breakpoints([(10, 0.002), (60, 0.008), (300, 0.008), (2000, 0.0005)], fsv)]),
           delimiter=",", header="freq [Hz],ASD [g2/Hz]", comments="", fmt="%.9g")
np.savetxt(os.path.join(OUT, "test_asd.csv"),
           np.column_stack([fsv, spectral.asd_from_breakpoints([(10, 0.01), (60, 0.04), (300, 0.04), (2000, 0.0025)], fsv)]),
           delimiter=",", header="freq [Hz],ASD [g2/Hz]", comments="", fmt="%.9g")
print("example data written to", OUT)

# 8. damping examples (vibtools 0.4.0)
fs4 = 10000.0
t4 = np.arange(0, 1.2, 1 / fs4)
ring = np.zeros_like(t4)
for fnm, zm, am in ((120.0, 0.015, 1.0), (410.0, 0.025, 0.5)):
    wn = 2 * np.pi * fnm
    ring += am * np.exp(-zm * wn * t4) * np.sin(wn * np.sqrt(1 - zm**2) * t4)
ring = np.concatenate([np.zeros(200), ring])[: t4.size] + 0.003 * rng.standard_normal(t4.size)
np.savetxt(os.path.join(OUT, "impact_ringdown.csv"), np.column_stack([t4, ring * 20]),
           delimiter=",", header="Time [s],response [g]", comments="", fmt="%.9g")
fp_, pp_ = spectral.psd_welch(resp[0], fs, df=0.25)
mm = (fp_ >= 20) & (fp_ <= 1000)
np.savetxt(os.path.join(OUT, "tip_response_psd.csv"), np.column_stack([fp_[mm], pp_[mm]]),
           delimiter=",", header="freq [Hz],tip PSD [g2/Hz]", comments="", fmt="%.6g")
fnc = np.array([100.0, 106.0]); zc = np.array([0.02, 0.03]); gc = np.array([0.8, 0.5])
respc = sdof.modal_response(base, fs, fnc, zc, gc, np.array([[1.0, 0.9]]), "abs_acc")[0]
respc += 0.03 * rng.standard_normal(respc.size)
np.savetxt(os.path.join(OUT, "shaker_close_modes.csv"), np.column_stack([t, base, respc]),
           delimiter=",", header="Time [s],control [g],bracket [g]", comments="", fmt="%.9g")
print("damping examples written")
