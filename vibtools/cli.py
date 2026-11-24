"""Command-line interface: `vibtools <command> ...` (also `python -m vibtools`).

Every command reads text/CSV tables (one header row, ANSYS Workbench exports work)
or UFF dataset 58 files, prints the key numbers, and writes a CSV and a PNG next to
the input (or into --out). Run `vibtools <command> -h` for the options.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap

import numpy as np

from . import __version__, fatigue, hav, io, modal, sdof, shock, spectral
from .fatigue import SN

G0 = 9.80665


# ------------------------------------------------------------------ helpers
def _load(path):
    """Return (columns dict, units dict). UFF files give x/y of the first block."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".uff", ".unv"):
        blocks = io.read_uff58(path)
        if not blocks:
            sys.exit(f"no dataset 58 blocks in {path}")
        b = blocks[0]
        y = b["y"]
        data = {"x": b["x"], "y": np.abs(y) if np.iscomplexobj(y) else y}
        return data, {"x": b.get("x_unit", ""), "y": b.get("y_unit", "")}
    return io.read_table(path)


def _col(data, name, fallback_index):
    keys = list(data)
    if name is None:
        return keys[fallback_index], data[keys[fallback_index]]
    if name in data:
        return name, data[name]
    for k in keys:
        if k.lower() == name.lower():
            return k, data[k]
    sys.exit(f"column '{name}' not found; available: {', '.join(keys)}")


def _time_and_fs(data, fs):
    keys = list(data)
    tkey = next((k for k in keys if k.lower() in ("time", "t", "time_s", "x")), None)
    if fs is None:
        if tkey is None:
            sys.exit("no time column found; give the sample rate with --fs")
        t = data[tkey]
        fs = (len(t) - 1) / (t[-1] - t[0])          # robust to rounded time stamps
    chans = [k for k in keys if k != tkey]
    return fs, chans


def _scale(units):
    return {"g": G0, "ms2": 1.0, "m/s2": 1.0, "mm/s2": 1e-3, "in/s2": 0.0254}[units.lower()]


def _out(args, stem, suffix):
    base = os.path.splitext(os.path.basename(stem))[0]
    d = args.out or (os.path.dirname(os.path.abspath(stem)) if os.path.exists(stem) else os.getcwd())
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{base}_{suffix}")


def _write_csv(path, header, cols):
    arr = np.column_stack(cols)
    np.savetxt(path, arr, delimiter=",", header=",".join(header), comments="", fmt="%.8g")
    print(f"  wrote {path}")


# The graphical interface sets this to receive each figure for live display.
PLOT_HOOK = None


def _plot(path, fn, grid=True):
    from matplotlib.figure import Figure      # no pyplot: works with any backend and in threads
    fig = Figure(figsize=(7, 3.6))
    ax = fig.subplots()
    fn(ax)
    if grid:
        ax.grid(True, which="both", color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"  wrote {path}")
    if PLOT_HOOK is not None:
        PLOT_HOOK(fig)


def _breakpoints(text):
    pts = []
    for tok in text.replace(";", " ").split():
        f, a = tok.split(",")
        pts.append((float(f), float(a)))
    return pts


def _asd_input(args):
    """ASD from --breakpoints or from a file (first column Hz, second g^2/Hz)."""
    if getattr(args, "breakpoints", None):
        bp = _breakpoints(args.breakpoints)
        f = np.linspace(bp[0][0], bp[-1][0], 40000)
        return f, spectral.asd_from_breakpoints(bp, f), "profile"
    if not getattr(args, "file", None):
        sys.exit("give an ASD file or --breakpoints \"f1,a1 f2,a2 ...\"")
    data, _ = _load(args.file)
    keys = list(data)
    return data[keys[0]], data[keys[1]], args.file


# ------------------------------------------------------------------ commands
def cmd_psd(args):
    data, _ = _load(args.file)
    fs, chans = _time_and_fs(data, args.fs)
    if args.channel:
        chans = [_col(data, args.channel, 1)[0]]
    print(f"sample rate {fs:.1f} Hz, resolution {args.df} Hz, band {args.band[0]}-{args.band[1]} Hz")
    cols, header = [], ["freq_Hz"]
    for c in chans:
        f, p = spectral.psd_welch(data[c] * _scale(args.units) / G0, fs, df=args.df)
        if not cols:
            cols.append(f)
        cols.append(p); header.append(f"{c}_g2_per_Hz")
        print(f"  {c:>20s}: {spectral.grms(f, p, *args.band):.4g} Grms "
              f"({len(data[c]) / fs / (1 / args.df) * 2 - 1:.0f} averages approx.)")
    out = _out(args, args.file, "psd")
    _write_csv(out + ".csv", header, cols)
    _plot(out + ".png", lambda ax: ([ax.loglog(cols[0][1:], c[1:], lw=1.4, label=h)
                                     for c, h in zip(cols[1:], header[1:])],
                                    ax.set(xlabel="Hz", ylabel="g²/Hz", xlim=args.band), ax.legend(fontsize=8)))


def cmd_frf(args):
    data, _ = _load(args.file)
    fs, _ = _time_and_fs(data, args.fs)
    ni, xin = _col(data, args.input, 1)
    no, yout = _col(data, args.output, 2)
    f, H1, H2, coh = spectral.frf_estimates(xin, yout, fs, df=args.df)
    m = (f >= args.band[0]) & (f <= args.band[1])
    good = m & (coh >= args.min_coherence)
    if not good.any():
        sys.exit(f"no frequency in the band has coherence >= {args.min_coherence}")
    k = np.argmax(np.abs(H1) * good)
    print(f"FRF {no} / {ni}: peak |H1| = {abs(H1[k]):.4g} at {f[k]:.2f} Hz, coherence {coh[k]:.3f}")
    try:
        hp = modal.half_power_damping(f, np.abs(H1), f_range=(f[k] * 0.8, f[k] * 1.2))
        print(f"  half-power: fn = {hp['fn']:.2f} Hz, zeta = {hp['zeta']:.4f}, Q = {hp['Q']:.1f}")
    except Exception as exc:                     # noqa: BLE001
        print(f"  half-power damping not available ({exc})")
    out = _out(args, args.file, "frf")
    _write_csv(out + ".csv", ["freq_Hz", "H1_re", "H1_im", "H2_re", "H2_im", "coherence"],
               [f, H1.real, H1.imag, H2.real, H2.imag, coh])

    def draw(ax):
        ax.semilogy(f[m], np.abs(H1[m]), lw=1.5, label="|H1|")
        ax.semilogy(f[m], np.abs(H2[m]), lw=1.0, ls="--", label="|H2|")
        ax2 = ax.twinx(); ax2.plot(f[m], coh[m], color="#9a9993", lw=0.8); ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("coherence")
        gm = good & np.isfinite(H1)
        lo, hi = np.abs(H1[gm]).min(), np.abs(H1[gm]).max()
        ax.set_ylim(lo / 3, hi * 3)                  # scale to the region with good coherence
        ax.set(xlabel="Hz", ylabel=f"|{no} / {ni}|",
               title=f"coherence ≥ {args.min_coherence} sets the scale; grey = coherence")
        ax.legend(fontsize=8, loc="upper right")
    _plot(out + ".png", draw)


def cmd_srs(args):
    data, _ = _load(args.file)
    fs, chans = _time_and_fs(data, args.fs)
    name, a = _col(data, args.channel, 1) if args.channel else (chans[0], data[chans[0]])
    a_ms2 = a * _scale(args.units)
    fr = shock.octave_freqs(args.fmin, args.fmax, args.n_per_oct)
    r = shock.srs(a_ms2, fs, fr, Q=args.q)
    dv = shock.velocity_change(a_ms2, fs)
    print(f"SRS of {name}, Q = {args.q}: peak input {np.abs(a_ms2).max() / G0:.3g} g, "
          f"velocity change {dv:.4g} m/s ({dv / 0.0254:.3g} in/s)")
    i = np.argmax(r.acc_max)
    print(f"  max SRS {r.acc_max[i] / G0:.4g} g at {fr[i]:.1f} Hz; max pseudo velocity "
          f"{r.pseudo_vel.max():.4g} m/s at {fr[np.argmax(r.pseudo_vel)]:.1f} Hz")
    out = _out(args, args.file, f"srs_{name}")
    _write_csv(out + ".csv", ["freq_Hz", "abs_acc_g", "rel_disp_m", "pseudo_vel_m_s"],
               [fr, r.acc_max / G0, r.rel_disp, r.pseudo_vel])
    _plot(out + ".png", lambda ax: (ax.loglog(fr, r.acc_max / G0, lw=1.6, label="absolute acceleration (g)"),
                                    ax.loglog(fr, r.pseudo_vel / 0.0254, lw=1.6, label="pseudo velocity (in/s)"),
                                    ax.set(xlabel="natural frequency (Hz)", title=f"SRS, Q = {args.q}"),
                                    ax.legend(fontsize=8)))


def cmd_profile(args):
    f, asd, src = _asd_input(args)
    band = args.band or (f[0], f[-1])
    r = spectral.velocity_from_asd(f, asd, *band)
    print(f"ASD {src}, band {band[0]:g}-{band[1]:g} Hz")
    print(f"  Grms (1 sigma)          {r['grms']:.4g} g")
    print(f"  velocity RMS (1 sigma)  {r['v_rms'] * 1e3:.4g} mm/s  ({r['v_rms'] / 0.0254:.3g} in/s)")
    print(f"  displacement RMS        {r['d_rms'] * 1e3:.4g} mm;  3-sigma peak-to-peak "
          f"{6 * r['d_rms'] * 1e3:.3g} mm")
    if args.fn:
        q = args.q
        print(f"  Miles at fn = {args.fn:g} Hz, Q = {q:g}: "
              f"{sdof.miles_grms(args.fn, q, float(np.interp(args.fn, f, asd))):.4g} Grms")
    out = _out(args, src if os.path.exists(str(src)) else "asd", "profile")
    _write_csv(out + ".csv", ["freq_Hz", "vel_psd_m2_s2_Hz", "disp_psd_m2_Hz"], [r["f"], r["v_psd"], r["d_psd"]])
    _plot(out + ".png", lambda ax: (ax.loglog(f, asd, lw=1.6), ax.set(xlabel="Hz", ylabel="g²/Hz",
                                                                     title=f"{r['grms']:.3g} Grms")))


def _sn(args):
    if args.sn_points:
        (s1, n1), (s2, n2) = (tuple(map(float, p.split(","))) for p in args.sn_points)
        return SN.from_points((s1, n1), (s2, n2))
    return SN(args.sn_c, args.sn_m)


def cmd_fatigue(args):
    data, units = _load(args.file)
    keys = list(data)
    f, G = data[keys[0]], data[keys[1]]
    sn = _sn(args)
    d = fatigue.dirlik_damage(f, G, args.time, sn)
    nb = fatigue.narrowband_damage(f, G, args.time, sn)
    print(f"stress PSD {args.file}: RMS {d['rms']:.4g}, peak rate {d['EP']:.4g} Hz, S-N m = {sn.m:.3g}")
    print(f"  Dirlik       D = {d['D']:.4g} over {args.time:g} s   life {d['life_s'] / 3600:.4g} h")
    print(f"  narrow-band  D = {nb:.4g}   (conservative for wide-band stress)")
    print("  damage scales as stress^m: report a range, not one number")
    out = _out(args, args.file, "fatigue")
    _plot(out + ".png", lambda ax: (ax.loglog(f[G > 0], G[G > 0], lw=1.6),
                                    ax.set(xlabel="Hz", ylabel="stress²/Hz",
                                           title=f"Stress PSD: RMS {d['rms']:.3g}, Dirlik life "
                                                 f"{d['life_s'] / 3600:.3g} h")))


def cmd_fds(args):
    f, asd, src = _asd_input(args)
    fns = shock.octave_freqs(args.fmin, args.fmax, 12)
    D = fatigue.fds_random(f, asd, args.time, fns, Q=args.q, b=args.b)
    cols, header = [fns, D], ["fn_Hz", "damage_rel"]
    print(f"FDS of {src}, {args.time:g} s, Q = {args.q:g}, b = {args.b:g} (relative units)")
    if args.compare:
        data, _ = _load(args.compare)
        k = list(data)
        D2 = fatigue.fds_random(data[k[0]], data[k[1]], args.compare_time or args.time, fns, Q=args.q, b=args.b)
        ratio = D / D2
        cols += [D2, ratio]; header += ["damage_compare", "ratio"]
        print(f"  ratio first/compare: min {ratio.min():.3g} at {fns[np.argmin(ratio)]:.0f} Hz, "
              f"max {ratio.max():.3g} at {fns[np.argmax(ratio)]:.0f} Hz "
              "(>= 1 everywhere means the first environment is at least as damaging)")
    out = _out(args, src if os.path.exists(str(src)) else "asd", "fds")
    _write_csv(out + ".csv", header, cols)
    _plot(out + ".png", lambda ax: ([ax.loglog(fns, c, lw=1.6, label=h) for c, h in
                                     zip(cols[1:3], header[1:3])], ax.set(xlabel="natural frequency (Hz)",
                                                                          ylabel="damage (relative)"),
                                    ax.legend(fontsize=8)))


def cmd_hav(args):
    data, _ = _load(args.file)
    fs, chans = _time_and_fs(data, args.fs)
    names = [args.x, args.y, args.z] if args.x else chans[:3]
    sc = _scale(args.units)
    vals, spectra = [], []
    for n in names:
        k, v = _col(data, n, 1)
        vals.append(hav.weighted_rms(v * sc, fs))
        fw, pw = spectral.psd_welch(hav.wh_weight(v * sc, fs), fs, df=1.0)
        spectra.append((k, fw, pw))
        print(f"  a_hw {k:>16s}: {vals[-1]:.4g} m/s²")
    ahv = hav.vibration_total_value(*vals)
    print(f"vibration total value a_hv = {ahv:.4g} m/s²")
    if args.hours:
        print(f"  A(8) for {args.hours:g} h = {hav.daily_exposure(ahv, args.hours):.4g} m/s²")
    print(f"  time to reach A(8) 2.5 m/s²: {hav.time_to_reach(ahv, 2.5):.3g} h;  "
          f"5 m/s²: {hav.time_to_reach(ahv, 5.0):.3g} h  (EU action / limit values; verify)")
    out = _out(args, args.file, "hav")
    _plot(out + ".png", lambda ax: ([ax.loglog(fw[1:], pw[1:], lw=1.3, label=f"{k}: a_hw {a:.3g} m/s²")
                                     for (k, fw, pw), a in zip(spectra, vals)],
                                    ax.set(xlim=(5, 1500), xlabel="Hz", ylabel="(m/s²)²/Hz, W_h weighted",
                                           title=f"Weighted handle spectra: a_hv = {ahv:.3g} m/s²"),
                                    ax.legend(fontsize=8)))


def cmd_mac(args):
    def read_modes(p):
        data, _ = io.read_table(p)
        k = list(data)
        return data[k[0]], np.column_stack([data[c] for c in k[1:]]).T     # freqs, (dof, mode)
    ft, pt = read_modes(args.test)
    fa, pa = read_modes(args.fea)
    if pt.shape[0] != pa.shape[0]:
        sys.exit(f"test has {pt.shape[0]} DOFs, FEA has {pa.shape[0]}: sample FEA at the test sensors")
    rows = modal.correlation_table(ft, fa, pt, pa)
    print(modal.format_table(rows))
    M = modal.mac(pt, pa)
    out = _out(args, args.test, "mac")
    np.savetxt(out + ".csv", M, delimiter=",", fmt="%.4f",
               header="MAC matrix: rows test modes, columns FEA modes", comments="")
    print(f"  wrote {out}.csv")

    def draw(ax):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="Blues")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.6 else "black")
        ax.set_xticks(range(M.shape[1]), [f"{x:.0f} Hz" for x in fa], fontsize=8)
        ax.set_yticks(range(M.shape[0]), [f"{x:.0f} Hz" for x in ft], fontsize=8)
        ax.set(xlabel="FEA modes", ylabel="test modes", title="MAC matrix")
        ax.figure.colorbar(im, ax=ax, shrink=0.8)
    _plot(out + ".png", draw, grid=False)


def cmd_damping(args):
    fn_arg = args.fn[0] if args.fn else None
    if args.method == "decay":
        data, _ = _load(args.file)
        fs, chans = _time_and_fs(data, args.fs)
        name, x = _col(data, args.channel, 1) if args.channel else (chans[0], data[chans[0]])
        r = modal.log_decrement(x, fs, fn=fn_arg, band=tuple(args.band) if args.band else None)
        print(f"Free decay of {name}: fn = {r['fn']:.3f} Hz, zeta = {r['zeta']:.4f} ({r['zeta'] * 100:.2f} %), "
              f"Q = {r['Q']:.1f}, log decrement = {r['delta']:.4f}, {r['n_peaks']} peaks")
        verdict = ("viscous-like (exponential decay)" if r["r2_log"] >= r["r2_linear"]
                   else "friction-like (linear decay): one zeta does not describe it; quote the amplitude range")
        print(f"  fit quality: R² ln(peaks) {r['r2_log']:.4f} vs R² linear {r['r2_linear']:.4f} -> {verdict}")
        out = _out(args, args.file, f"decay_{name}_{r['fn']:.0f}Hz")
        tp, ap = r["t_peaks"], r["a_peaks"]
        s_, c_ = np.polyfit(tp, np.log(ap), 1)
        _write_csv(out + ".csv", ["t_peak_s", "peak_amplitude"], [tp, ap])
        _plot(out + ".png", lambda ax: (ax.semilogy(tp, ap, "o", ms=5, label="filtered peaks"),
                                        ax.semilogy(tp, np.exp(s_ * tp + c_), lw=1.6,
                                                    label=f"exponential fit, ζ = {r['zeta'] * 100:.2f} %"),
                                        ax.set(xlabel="time (s)", ylabel="peak amplitude",
                                               title=f"Log decrement, mode at {r['fn']:.1f} Hz"),
                                        ax.legend(fontsize=8)))
    elif args.method == "psd":
        data, _ = _load(args.file)
        k = list(data)
        f, p = data[k[0]], data[k[1]]
        band = tuple(args.band) if args.band else ((0.7 * fn_arg, 1.3 * fn_arg) if fn_arg else None)
        r = modal.half_power_damping_psd(f, p, band)
        print(f"Response PSD half-power (points at half the PSD peak, -3 dB): fn = {r['fn']:.3f} Hz, "
              f"zeta = {r['zeta']:.4f} ({r['zeta'] * 100:.2f} %), Q = {r['Q']:.1f}")
        print("  valid for one well-separated mode on a roughly flat input; check that the peak spans "
              "at least 5 frequency lines")
        out = _out(args, args.file, "psd_damping")
        m = (f >= r["f1"] - 3 * (r["f2"] - r["f1"])) & (f <= r["f2"] + 3 * (r["f2"] - r["f1"]))
        pk = np.interp(r["fn"], f, p)
        _plot(out + ".png", lambda ax: (ax.plot(f[m], p[m], lw=1.6), ax.axhline(pk / 2, color="#9a9993", ls="--"),
                                        ax.axvspan(r["f1"], r["f2"], color="#2a78d6", alpha=0.12),
                                        ax.set(xlabel="Hz", ylabel="PSD",
                                               title=f"Half-power at PSD/2: ζ = {r['zeta'] * 100:.2f} %")))
    else:
        data, _ = _load(args.file)
        fs, _ = _time_and_fs(data, args.fs)
        ni, xin = _col(data, args.input, 1)
        no, yout = _col(data, args.output, 2)
        if not args.fn:
            sys.exit("give one starting frequency per mode with --fn, e.g. --fn 100 106")
        f, H1, _, coh = spectral.frf_estimates(xin, yout, fs, df=args.df)
        r = modal.fit_modes(f, H1, args.fn, f_range=tuple(args.band) if args.band else None)
        print(f"Multi-mode fit of {no}/{ni} (H1), fit error {r['rms_error_pct']:.2f} %:")
        print("  mode     fn (Hz)    zeta      Q")
        for i, (a, z) in enumerate(zip(r["fn"], r["zeta"])):
            print(f"  {i + 1:4d}  {a:9.3f}  {z:7.4f}  {1 / (2 * z):6.1f}")
        lo, hi = args.band if args.band else (0.8 * min(args.fn), 1.2 * max(args.fn))
        m = (f >= lo) & (f <= hi)
        try:
            hp = modal.half_power_damping(f[m], np.abs(H1[m]))
            print(f"  for comparison, half-power on the highest peak: fn = {hp['fn']:.2f} Hz, zeta = {hp['zeta']:.4f}"
                  " (biased when modes overlap)")
        except ValueError:
            pass
        out = _out(args, args.file, "frf_fit")
        _write_csv(out + ".csv", ["freq_Hz", "H1_mag", "fit_mag", "coherence"],
                   [f[m], np.abs(H1[m]), np.abs(r["fit"][m]), coh[m]])
        _plot(out + ".png", lambda ax: (ax.semilogy(f[m], np.abs(H1[m]), lw=1.2, label="measured |H1|"),
                                        ax.semilogy(f[m], np.abs(r["fit"][m]), lw=1.8, ls="--",
                                                    label=f"{len(args.fn)}-mode fit"),
                                        ax.set(xlabel="Hz", ylabel=f"|{no} / {ni}|", title="Multi-mode curve fit"),
                                        ax.legend(fontsize=8)))


def cmd_uff2csv(args):
    blocks = io.read_uff58(args.file)
    for i, b in enumerate(blocks):
        out = _out(args, args.file, f"block{i + 1}")
        y = b["y"]
        if np.iscomplexobj(y):
            _write_csv(out + ".csv", [b["x_label"] or "x", "re", "im", "mag", "phase_deg"],
                       [b["x"], y.real, y.imag, np.abs(y), np.degrees(np.angle(y))])
        else:
            _write_csv(out + ".csv", [b["x_label"] or "x", b["y_label"] or "y"], [b["x"], y])
        print(f"    block {i + 1}: {b['func_name']}, node {b['response_node']} dir {b['response_dir']}, "
              f"{len(b['x'])} points")


def cmd_demo(args):
    from .examples_runner import run_demo
    run_demo(args.out or os.getcwd())


def cmd_benchmark(args):
    from .examples_runner import run_benchmark
    run_benchmark(args.out or os.getcwd())


def cmd_selftest(args):
    """Quick validation of the core numbers (a subset of the pytest suite)."""
    ok = True

    def check(name, value, ref, rel):
        nonlocal ok
        good = abs(value / ref - 1) <= rel
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name}: {value:.5g} (reference {ref:.5g}, tol {rel:.1%})")

    fs = 20000.0
    t, a = shock.half_sine(10 * 386.09, 0.011, fs, tail=0.2)
    r = shock.srs(a, fs, np.array([10.0, 30.0, 60.0, 100.0, 300.0]), Q=10)
    check("SRS of 10 g 11 ms half-sine, max (Irvine 16.5 g)", r.acc_max.max() / 386.09, 16.5, 0.05)
    check("pseudo velocity max (Irvine 25.2 in/s)", r.pseudo_vel.max(), 25.2, 0.05)
    check("half-sine velocity change (27 in/s)", shock.velocity_change(a, fs), 27.0, 0.01)
    check("W_h at 31.5 Hz (ISO 5349-1: 0.519)", abs(hav.wh_response(1000 * 10 ** (-15 / 10))), 0.519, 0.005)
    check("W_h at 250 Hz (ISO 5349-1: 0.0634)", abs(hav.wh_response(1000 * 10 ** (-6 / 10))), 0.0634, 0.005)
    f = np.linspace(20, 2000, 400001)
    v = spectral.velocity_from_asd(f, np.full_like(f, 0.04), g=9.81)["v_rms"]
    check("velocity RMS of flat 0.04 g²/Hz", v, np.sqrt(0.04 * 9.81 ** 2 * (1 / 20 - 1 / 2000)) / (2 * np.pi), 1e-3)
    check("Miles, fn 200 Hz, Q 20, 0.04 g²/Hz", sdof.miles_grms(200, 20, 0.04),
          np.sqrt(np.pi / 2 * 200 * 20 * 0.04), 1e-9)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


# ------------------------------------------------------------------ parser
def build_parser():
    p = argparse.ArgumentParser(
        prog="vibtools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""\
            vibtools {__version__}: shock and vibration test-data processing and FEA correlation.

            Input tables: text/CSV with one header row (ANSYS Workbench exports work), or
            UFF dataset 58 (.uff/.unv). Time histories: a 'Time' column or --fs.
            Each command prints its results and writes <input>_<command>.csv/.png."""),
        epilog="Examples:\n"
               "  vibtools psd shaker_run.csv --df 1 --band 20 2000\n"
               "  vibtools frf shaker_run.csv --input control --output tip\n"
               "  vibtools srs drop.csv --units g --q 10\n"
               "  vibtools profile --breakpoints \"20,0.01 80,0.04 500,0.04 2000,0.01\"\n"
               "  vibtools fatigue stress_psd.txt --time 3600 --sn-points 250,1e4 100,1e7\n"
               "  vibtools hav handle.csv --units ms2 --hours 2\n"
               "  vibtools mac test_modes.csv fea_modes.csv\n"
               "  vibtools selftest")
    p.add_argument("--version", action="version", version=f"vibtools {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="command")

    def add(name, fn, help_):
        s = sub.add_parser(name, help=help_, description=help_)
        s.set_defaults(func=fn)
        s.add_argument("--out", help="output folder (default: next to the input)")
        return s

    s = add("psd", cmd_psd, "PSD and Grms of time-history channels (Welch, Hann, 50%% overlap)")
    s.add_argument("file"); s.add_argument("--channel"); s.add_argument("--fs", type=float)
    s.add_argument("--df", type=float, default=1.0, help="frequency resolution, Hz (default 1)")
    s.add_argument("--band", type=float, nargs=2, default=(20.0, 2000.0), metavar=("F1", "F2"))
    s.add_argument("--units", default="g", choices=["g", "ms2", "mm/s2", "in/s2"])

    s = add("frf", cmd_frf, "H1, H2 and coherence between an input (control) and an output channel")
    s.add_argument("file"); s.add_argument("--input"); s.add_argument("--output"); s.add_argument("--fs", type=float)
    s.add_argument("--df", type=float, default=0.5)
    s.add_argument("--band", type=float, nargs=2, default=(5.0, 2000.0), metavar=("F1", "F2"))
    s.add_argument("--min-coherence", type=float, default=0.8, help="peak search only where coherence >= this")

    s = add("srs", cmd_srs, "shock response spectrum (abs. acceleration and pseudo velocity) and velocity change")
    s.add_argument("file"); s.add_argument("--channel"); s.add_argument("--fs", type=float)
    s.add_argument("--q", type=float, default=10.0); s.add_argument("--fmin", type=float, default=10.0)
    s.add_argument("--fmax", type=float, default=2000.0); s.add_argument("--n-per-oct", type=int, default=12)
    s.add_argument("--units", default="g", choices=["g", "ms2", "mm/s2", "in/s2"])

    s = add("profile", cmd_profile, "Grms, table velocity and displacement RMS of an ASD (file or breakpoints)")
    s.add_argument("file", nargs="?"); s.add_argument("--breakpoints", help='"f1,asd1 f2,asd2 ..." in Hz, g²/Hz')
    s.add_argument("--band", type=float, nargs=2, metavar=("F1", "F2"))
    s.add_argument("--fn", type=float, help="also give Miles' Grms for a mode at fn"); s.add_argument("--q", type=float, default=10.0)

    s = add("fatigue", cmd_fatigue, "Dirlik and narrow-band damage from a stress PSD (columns: Hz, stress²/Hz)")
    s.add_argument("file"); s.add_argument("--time", type=float, required=True, help="exposure, s")
    s.add_argument("--sn-points", nargs=2, metavar=("S1,N1", "S2,N2"), help="two S-N points (amplitude, cycles)")
    s.add_argument("--sn-c", type=float, help="S-N constant C in N*S^m = C"); s.add_argument("--sn-m", type=float)

    s = add("fds", cmd_fds, "fatigue damage spectrum of an ASD; optional comparison with a second ASD")
    s.add_argument("file", nargs="?"); s.add_argument("--breakpoints")
    s.add_argument("--time", type=float, required=True); s.add_argument("--q", type=float, default=10.0)
    s.add_argument("--b", type=float, default=8.0); s.add_argument("--fmin", type=float, default=10.0)
    s.add_argument("--fmax", type=float, default=2000.0); s.add_argument("--compare")
    s.add_argument("--compare-time", type=float)

    s = add("hav", cmd_hav, "hand-arm vibration: W_h weighted RMS per axis, a_hv, A(8)")
    s.add_argument("file"); s.add_argument("--x"); s.add_argument("--y"); s.add_argument("--z")
    s.add_argument("--fs", type=float); s.add_argument("--hours", type=float)
    s.add_argument("--units", default="ms2", choices=["g", "ms2", "mm/s2", "in/s2"])

    s = add("mac", cmd_mac, "modal correlation table (frequency error, MAC); rows = modes: freq, dof1, dof2 ...")
    s.add_argument("test"); s.add_argument("fea")

    s = add("uff2csv", cmd_uff2csv, "convert every dataset-58 block of a UFF file to CSV")
    s.add_argument("file")

    s = add("damping", cmd_damping, "measure damping: free decay (log decrement), response PSD "
                                    "(half-power at -3 dB) or multi-mode FRF curve fit")
    s.add_argument("method", choices=["decay", "psd", "frf"])
    s.add_argument("file")
    s.add_argument("--channel", help="decay: the ring-down channel")
    s.add_argument("--input", help="frf: control (input) channel"); s.add_argument("--output", help="frf: response channel")
    s.add_argument("--fn", type=float, nargs="+", help="decay: mode frequency; frf: one start value per mode")
    s.add_argument("--band", type=float, nargs=2, metavar=("F1", "F2"), help="frequency band around the mode(s)")
    s.add_argument("--fs", type=float); s.add_argument("--df", type=float, default=0.25)

    add("demo", cmd_demo, "run the synthetic test-vs-FEA demo and write demo_output.png")
    add("benchmark", cmd_benchmark, "print the reference values of the cantilever FEA benchmark")
    add("selftest", cmd_selftest, "quick validation of core numbers against published values")
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    if not argv:
        parser.print_help()
        if getattr(sys, "frozen", False) and sys.stdin and sys.stdin.isatty():
            input("\nThis is a command-line tool. Open a terminal in this folder and type, for example,\n"
                  "  vibtools selftest\nPress Enter to close.")
        return 0
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help(); return 0
    rc = args.func(args)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
