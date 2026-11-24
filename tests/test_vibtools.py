"""Validation tests: analytic results and values published in Irvine (rev. Q)."""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from vibtools import SN, fatigue, handcalc, mac, modal, sdof, shock, spectral


# ---------------- SDOF / SRS ----------------
def test_filter_matches_analytic_resonance_gain():
    fs, fn, zeta = 20000.0, 100.0, 0.02
    t = np.arange(0, 3.0, 1 / fs)
    y = sdof.sdof_response(np.sin(2 * np.pi * fn * t), fs, fn, 1 / (2 * zeta))
    amp = np.abs(y[int(2.5 * fs):]).max()
    expected = abs(sdof.transmissibility(fn, fn, zeta))
    assert amp == pytest.approx(expected, rel=0.02)


def test_srs_half_sine_matches_irvine_book():
    # Irvine rev. Q, Section 32: 10 G, 11 ms half-sine, Q=10, 1-1000 Hz
    # gives max accel 16.5 G and max pseudo velocity 25.2 in/s.
    fs = 20000.0
    _, a = shock.half_sine(10.0, 0.011, fs)
    f = shock.octave_freqs(1.0, 1000.0, 12)
    r = shock.srs(a * 386.09, fs, f, Q=10)          # accel in in/s^2
    assert r.acc_max.max() / 386.09 == pytest.approx(16.5, rel=0.05)
    assert r.pseudo_vel.max() == pytest.approx(25.2, rel=0.05)


def test_velocity_change_half_sine():
    fs = 50000.0
    _, a = shock.half_sine(10.0 * 386.09, 0.011, fs)
    assert shock.velocity_change(a, fs) == pytest.approx(27.0, rel=0.01)  # in/s


def test_modal_abs_acc_matches_modal_transmissibility_at_sine():
    fs = 20000.0
    fn, zeta = np.array([80.0, 210.0]), np.array([0.02, 0.03])
    gamma, shapes = np.array([0.8, 0.3]), np.array([[1.0, -0.6]])
    f0 = 95.0
    t = np.arange(0, 4.0, 1 / fs)
    y = sdof.modal_response(np.sin(2 * np.pi * f0 * t), fs, fn, zeta, gamma, shapes, "abs_acc")[0]
    amp = np.abs(y[int(3.0 * fs):]).max()
    T = np.abs(sdof.modal_transmissibility(f0, fn, zeta, gamma, shapes))[0, 0]
    assert amp == pytest.approx(T, rel=0.02)


# ---------------- spectra ----------------
def test_grms_of_white_noise():
    fs, asd = 8192.0, 0.04
    f_t = np.linspace(0, fs / 2, 3000)
    _, x = spectral.synth_gaussian(f_t, np.full_like(f_t, asd), 60.0, fs, seed=1)
    f, p = spectral.psd_welch(x, fs, df=2.0)
    assert spectral.grms(f, p) == pytest.approx(np.sqrt(asd * fs / 2), rel=0.03)


def test_breakpoint_profile_grms():
    # 20-80 Hz at +3 dB/oct, 80-350 flat 0.04, 350-2000 -3 dB/oct (classic profile)
    bp = [(20, 0.01), (80, 0.04), (350, 0.04), (2000, 0.04 * (350 / 2000) ** 1.0)]
    assert spectral.grms_from_breakpoints(bp) == pytest.approx(np.sqrt(
        trapezoid(spectral.asd_from_breakpoints(bp, np.linspace(20, 2000, 20000)),
                     np.linspace(20, 2000, 20000))), rel=1e-6)


def test_velocity_from_flat_asd_matches_closed_form():
    # flat ASD A between f1 and f2: v_rms^2 = A g^2 (1/f1 - 1/f2) / (4 pi^2)
    A, f1, f2, g = 0.04, 20.0, 2000.0, 9.81
    f = np.linspace(f1, f2, 400001)
    r = spectral.velocity_from_asd(f, np.full_like(f, A), g=g)
    v = np.sqrt(A * g**2 * (1 / f1 - 1 / f2) / (4 * np.pi**2))
    d = np.sqrt(A * g**2 * (1 / f1**3 - 1 / f2**3) / (3 * 16 * np.pi**4))
    assert r["v_rms"] == pytest.approx(v, rel=1e-4)
    assert r["d_rms"] == pytest.approx(d, rel=1e-4)
    assert r["grms"] == pytest.approx(np.sqrt(A * (f2 - f1)), rel=1e-6)


def test_velocity_from_narrow_band_is_a_over_omega():
    # a narrow band around f0 behaves like a sine: v_rms = a_rms / (2 pi f0)
    f0, bw, A = 200.0, 2.0, 0.5
    f = np.linspace(f0 - bw / 2, f0 + bw / 2, 2001)
    r = spectral.velocity_from_asd(f, np.full_like(f, A))
    a_rms = np.sqrt(A * bw) * 9.81
    assert r["v_rms"] == pytest.approx(a_rms / (2 * np.pi * f0), rel=1e-3)


def test_h1_recovers_transmissibility():
    fs = 4096.0
    f_t = np.linspace(0, fs / 2, 2000)
    _, x = spectral.synth_gaussian(f_t, np.where(f_t > 10, 1.0, 0), 120.0, fs, seed=3)
    y = sdof.sdof_response(x, fs, 150.0, 25.0)
    f, H1, H2, coh = spectral.frf_estimates(x, y, fs, df=0.5)
    i = np.argmin(np.abs(f - 150.0))
    assert abs(H1[i]) == pytest.approx(abs(sdof.transmissibility(150.0, 150.0, 0.02)), rel=0.05)
    assert coh[i] > 0.98


# ---------------- modal ----------------
def test_half_power_damping_and_fit():
    f = np.linspace(100, 200, 2001)
    T = sdof.transmissibility(f, 150.0, 0.02)
    hp = modal.half_power_damping(f, np.abs(T), (120, 180))
    assert hp["fn"] == pytest.approx(150.0, rel=0.005)
    assert hp["zeta"] == pytest.approx(0.02, rel=0.05)
    rng = np.random.default_rng(0)
    noisy = np.abs(T) * (1 + 0.03 * rng.standard_normal(f.size))
    ft = modal.fit_transmissibility(f, noisy, 148.0, 0.03)
    assert ft["zeta"] == pytest.approx(0.02, rel=0.1)


def test_mac_and_pairing():
    A = np.array([[1, 0.2], [0.8, -0.7], [0.3, 1.0]])
    M = mac(A, A[:, ::-1])
    assert np.allclose(np.diag(mac(A, A)), 1.0)
    assert modal.pair_modes(M)[0][:2] in [(0, 1), (1, 0)]
    assert modal.freq_error_pct(100.0, 105.0) == pytest.approx(5.0)


# ---------------- fatigue ----------------
def test_rainflow_simple_and_sine():
    r, _, c = fatigue.rainflow([0, 10, 0, 10, 0])
    assert c.sum() == pytest.approx(2.0)
    assert np.all(r == 10)
    t = np.linspace(0, 20 * 2 * np.pi, 20000)
    r, _, c = fatigue.rainflow(5 * np.sin(t))
    assert c.sum() == pytest.approx(20, abs=1)
    assert r.max() == pytest.approx(10, rel=1e-3)


def test_sweep_cycle_density_and_damage():
    sn = SN.from_points((200.0, 1e5), (100.0, 1e7))
    f = np.linspace(10, 100, 500)
    D = fatigue.sweep_damage(f, np.full_like(f, 50.0), sn, rate=1.0, kind="log")
    cycles = 60 / np.log(2) * 90
    assert D == pytest.approx(cycles / sn.cycles(50.0), rel=1e-6)


def test_dirlik_narrowband_limit_and_steinberg_scale():
    sn = SN.from_points((200.0, 1e5), (100.0, 1e7))
    f = np.linspace(1, 400, 40000)
    zeta, fn = 0.01, 150.0
    G = np.abs(sdof.transmissibility(f, fn, zeta)) ** 2          # white input -> stress PSD
    G *= 25.0**2 / trapezoid(G, f)                              # 25 MPa rms
    T = 3600.0
    d_dirlik = fatigue.dirlik_damage(f, G, T, sn)["D"]
    d_nb = fatigue.narrowband_damage(f, G, T, sn)
    assert d_dirlik == pytest.approx(d_nb, rel=0.15)
    d_st = fatigue.steinberg_damage(25.0, fn, T, sn)
    assert d_st == pytest.approx(d_nb, rel=0.5)                   # screening-level agreement


@pytest.mark.slow
def test_dirlik_vs_rainflow_montecarlo_two_modes():
    sn = SN.from_points((200.0, 1e5), (100.0, 1e7))
    fs = 2048.0
    f = np.linspace(1, 900, 6000)
    G = (np.abs(sdof.transmissibility(f, 90.0, 0.03)) ** 2
         + 0.6 * np.abs(sdof.transmissibility(f, 260.0, 0.03)) ** 2)
    G *= 30.0**2 / trapezoid(G, f)
    T = 300.0
    _, x = spectral.synth_gaussian(f, G, T, fs, seed=7)
    d_mc = fatigue.rainflow_damage(x, sn)
    d_dk = fatigue.dirlik_damage(f, G, T, sn)["D"]
    assert d_dk == pytest.approx(d_mc, rel=0.15)


def test_equivalent_test_time():
    assert fatigue.equivalent_test_time(100.0, 0.01, 0.04, 6) == pytest.approx(100 * (0.25) ** 3)


# ---------------- hand calcs ----------------
def test_irvine_drop_example():
    # Section 41: K=5, aluminium rho*c = 51.8 lbf s/in^3, v = 167 in/s -> 43.3 ksi
    s = handcalc.stress_from_velocity(5.0, 51.8, 167.0)
    assert s == pytest.approx(43_253, rel=0.01)       # psi
    d = handcalc.drop_sdof(36 * 0.0254, 200.0)
    assert d["v"] / 0.0254 == pytest.approx(167, rel=0.01)
    assert d["accel_g"] == pytest.approx(543, rel=0.02)


def test_quick_check_runs():
    out = handcalc.quick_check("Al 6061-T6", 2.0, 50.0, 150.0, sigma_fea=18e6)
    assert out["sigma"] == pytest.approx(14.5e6, rel=0.05)
    assert 1.0 < out["ratio_fea_to_hand"] < 1.5


# ---------------- hand-arm vibration (0.3.0) ----------------
def test_wh_matches_iso5349_table():
    from vibtools import hav
    table = {8: 0.873, 16: 0.896, 31.5: 0.519, 63: 0.256, 125: 0.127, 250: 0.0634, 500: 0.0314,
             1000: 0.0135}
    for fc, w in table.items():
        f = 1000 * 10 ** (round(10 * np.log10(fc / 1000)) / 10)     # exact band centre
        assert abs(hav.wh_response(f)) == pytest.approx(w, rel=0.005)


def test_wh_weighted_rms_of_sine_and_a8():
    from vibtools import hav
    fs, A, f0 = 10000.0, 10.0, 100.0
    t = np.arange(0, 10.0, 1 / fs)
    x = A * np.sin(2 * np.pi * f0 * t)
    assert hav.weighted_rms(x, fs) == pytest.approx(A / np.sqrt(2) * abs(hav.wh_response(f0)), rel=0.01)
    assert hav.daily_exposure(4.0, 2.0) == pytest.approx(2.0)
    assert hav.daily_exposure([4.0, 3.0], [2.0, 8.0]) == pytest.approx(np.sqrt((16 * 2 + 9 * 8) / 8))
    assert hav.time_to_reach(5.0, 2.5) == pytest.approx(2.0)
    assert hav.vibration_total_value(3.0, 4.0, 12.0) == pytest.approx(13.0)


# ---------------- fatigue damage spectrum (0.3.0) ----------------
def test_fds_time_matches_random_and_scales():
    fs, T = 4096.0, 300.0
    f = np.linspace(0.5, fs / 2, 20000)
    asd = spectral.asd_from_breakpoints([(20, 0.04), (1000, 0.04)], f)
    _, x = spectral.synth_gaussian(f, asd, T, fs, seed=3)          # g
    fns = np.array([60.0, 200.0])
    d_t = fatigue.fds_time(x * 9.81, fs, fns, Q=10, b=4)
    d_r = fatigue.fds_random(f, asd, T, fns, Q=10, b=4)
    assert np.all(np.abs(d_t / d_r - 1) < 0.2)
    d_r4 = fatigue.fds_random(f, 4 * asd, T, fns, Q=10, b=4)
    assert d_r4 / d_r == pytest.approx(np.full(2, 4 ** 2), rel=1e-9)


# ---------------- file io (0.3.0) ----------------
def test_uff58_roundtrip_complex(tmp_path):
    from vibtools import io
    f = np.linspace(0, 500, 501)
    H = sdof.transmissibility(f, 120.0, 0.03)
    p = tmp_path / "frf.uff"
    io.write_uff58(p, f, H, func_type=5, response=(12, 3), y_label="Transmissibility", y_unit="g/g")
    blk = io.read_uff58(p)[0]
    assert blk["func_name"] == "transmissibility" and blk["response_node"] == 12
    assert blk["y_label"] == "Transmissibility" and blk["x_unit"] == "Hz"
    assert np.allclose(blk["x"], f) and np.allclose(blk["y"], H, rtol=1e-9)


def test_read_table_workbench_style(tmp_path):
    from vibtools import io
    p = tmp_path / "export.txt"
    p.write_text("Response PSD\nTabular Data\nFrequency [Hz]\tAcceleration [(mm/s²)²/Hz]\n"
                 "10.\t1.5e-002\n20.\t3.0e-002\n40.\t2.5e-002\n", encoding="utf-8")
    data, units = io.read_table(p)
    assert units["Frequency"] == "Hz"
    assert np.allclose(data["Frequency"], [10, 20, 40])
    assert np.allclose(data["Acceleration"], [0.015, 0.03, 0.025])


# ---------------- command-line interface (0.3.0) ----------------
def test_cli_selftest_and_profile(tmp_path, capsys):
    from vibtools.cli import main
    assert main(["selftest"]) == 0
    assert main(["profile", "--breakpoints", "20,0.01 80,0.04 500,0.04 2000,0.01", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ALL PASS" in out and "6.78" in out and "52.9" in out
    assert (tmp_path / "asd_profile.csv").exists()


# ---------------- damping measurement (0.4.0) ----------------
def test_log_decrement_two_mode_ringdown_and_coulomb_flag():
    fs = 10000.0
    t = np.arange(0, 1.5, 1 / fs)
    rng = np.random.default_rng(1)

    def mode(fn, z, a):
        wn = 2 * np.pi * fn
        return a * np.exp(-z * wn * t) * np.sin(wn * np.sqrt(1 - z**2) * t)
    x = mode(100.0, 0.02, 1.0) + mode(350.0, 0.03, 0.6) + 0.002 * rng.standard_normal(t.size)
    r = modal.log_decrement(x, fs, fn=100.0)
    assert r["zeta"] == pytest.approx(0.02, rel=0.05)
    assert r["fn"] == pytest.approx(100.0, rel=0.005)
    assert r["r2_log"] > r["r2_linear"]
    # friction (Coulomb) decay: amplitude falls linearly with time
    a = np.clip(1.0 - 1.2 * t, 0, None)
    xc = a * np.sin(2 * np.pi * 80.0 * t)
    rc = modal.log_decrement(xc, fs, fn=80.0, min_ratio=0.2)
    assert rc["r2_linear"] > rc["r2_log"]


def test_half_power_on_psd_uses_minus_3db():
    f = np.linspace(50, 150, 20001)
    psd = np.abs(sdof.transmissibility(f, 100.0, 0.02)) ** 2 * 0.04
    r = modal.half_power_damping_psd(f, psd)
    assert r["zeta"] == pytest.approx(0.02, rel=0.03)
    naive = modal.half_power_damping(f, psd)          # 0.707 rule applied to a PSD: wrong
    assert naive["zeta"] < 0.75 * 0.02


def test_fit_modes_resolves_close_modes():
    f = np.linspace(60, 160, 4001)
    fn = np.array([100.0, 106.0]); z = np.array([0.02, 0.03])
    gam = np.array([0.8, 0.5]); phi = np.array([[1.0, 0.9]])
    T = sdof.modal_transmissibility(f, fn, z, gam, phi)[0]
    rng = np.random.default_rng(2)
    Tn = T * (1 + 0.01 * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)))
    r = modal.fit_modes(f, Tn, [98.0, 108.0], f_range=(80, 130))
    assert r["fn"] == pytest.approx(fn, rel=0.005)
    assert r["zeta"] == pytest.approx(z, rel=0.08)
    assert r["rms_error_pct"] < 3
