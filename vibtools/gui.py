"""Graphical interface for vibtools (tkinter). Start with `python -m vibtools.gui` or vibtools_gui.exe.

Each tab fills in the same options as the command line, runs the same tested code in a
background thread, shows the printed results and an interactive plot (zoom, pan, save),
and writes CSV/PNG files to the output folder.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from . import __version__, cli

NAVY = "#1F3864"

# ------------------------------------------------------------------ tab specifications
# field: (key, label, kind, default, extra)
#   kind: file | savefile-less | col | float | int | text | choice | check
TABS = [
    ("PSD", "psd", "PSD and Grms of time-history channels (Welch, Hann, 50 % overlap).", [
        ("file", "Time-history file", "file", "", None),
        ("channel", "Channel", "col", "(all)", "file"),
        ("units", "Acceleration units", "choice", "g", ["g", "ms2", "mm/s2", "in/s2"]),
        ("df", "Resolution Δf (Hz)", "float", "1.0", None),
        ("band0", "Band from (Hz)", "float", "20", None),
        ("band1", "Band to (Hz)", "float", "2000", None),
        ("fs", "Sample rate (Hz), if no Time column", "float", "", None),
    ]),
    ("FRF", "frf", "Transmissibility H1, H2, coherence and half-power damping between two channels.", [
        ("file", "Time-history file", "file", "", None),
        ("input", "Input (control) channel", "col", "", "file"),
        ("output", "Output (response) channel", "col", "", "file"),
        ("df", "Resolution Δf (Hz)", "float", "0.5", None),
        ("band0", "Band from (Hz)", "float", "5", None),
        ("band1", "Band to (Hz)", "float", "2000", None),
        ("min_coherence", "Peak search: min. coherence", "float", "0.8", None),
        ("fs", "Sample rate (Hz), if no Time column", "float", "", None),
    ]),
    ("SRS", "srs", "Shock response spectrum, pseudo velocity and velocity change of a pulse.", [
        ("file", "Time-history file", "file", "", None),
        ("channel", "Channel", "col", "", "file"),
        ("units", "Acceleration units", "choice", "g", ["g", "ms2", "mm/s2", "in/s2"]),
        ("q", "Q", "float", "10", None),
        ("fmin", "Lowest frequency (Hz)", "float", "10", None),
        ("fmax", "Highest frequency (Hz)", "float", "2000", None),
        ("n_per_oct", "Points per octave", "int", "12", None),
        ("fs", "Sample rate (Hz), if no Time column", "float", "", None),
    ]),
    ("Profile", "profile", "Grms, table velocity and displacement RMS of an ASD; Miles' equation.\n"
     "Give breakpoints, or leave them empty and choose an ASD file (Hz, g²/Hz).", [
        ("breakpoints", "Breakpoints  f,ASD  f,ASD ...", "text", "20,0.01 80,0.04 500,0.04 2000,0.01", None),
        ("file", "or ASD file", "file", "", None),
        ("band0", "Band from (Hz), optional", "float", "", None),
        ("band1", "Band to (Hz), optional", "float", "", None),
        ("fn", "Miles: natural frequency (Hz), optional", "float", "", None),
        ("q", "Miles: Q", "float", "25", None),
    ]),
    ("Fatigue", "fatigue", "Dirlik and narrow-band damage from a stress PSD (Hz, stress²/Hz).", [
        ("file", "Stress PSD file", "file", "", None),
        ("time", "Exposure time (s)", "float", "3600", None),
        ("s1", "S-N point 1: stress amplitude", "float", "250", None),
        ("n1", "S-N point 1: cycles", "float", "1e4", None),
        ("s2", "S-N point 2: stress amplitude", "float", "100", None),
        ("n2", "S-N point 2: cycles", "float", "1e7", None),
    ]),
    ("FDS", "fds", "Fatigue damage spectrum of an ASD; optional comparison with a second ASD\n"
     "(ratio ≥ 1 everywhere: the first environment is at least as damaging).", [
        ("breakpoints", "Breakpoints  f,ASD ... (or a file below)", "text", "", None),
        ("file", "ASD file", "file", "", None),
        ("time", "Duration (s)", "float", "1800", None),
        ("compare", "Compare with ASD file (optional)", "file", "", None),
        ("compare_time", "Comparison duration (s)", "float", "", None),
        ("q", "Q", "float", "10", None),
        ("b", "Damage exponent b", "float", "8", None),
        ("fmin", "Lowest natural frequency (Hz)", "float", "10", None),
        ("fmax", "Highest natural frequency (Hz)", "float", "2000", None),
    ]),
    ("HAV", "hav", "Hand-arm vibration: W_h weighted RMS per axis, a_hv and A(8) (ISO 5349-1).", [
        ("file", "Handle time-history file", "file", "", None),
        ("x", "X channel", "col", "", "file"),
        ("y", "Y channel", "col", "", "file"),
        ("z", "Z channel", "col", "", "file"),
        ("units", "Acceleration units", "choice", "ms2", ["ms2", "g", "mm/s2", "in/s2"]),
        ("hours", "Daily exposure (h), optional", "float", "", None),
        ("fs", "Sample rate (Hz), if no Time column", "float", "", None),
    ]),
    ("MAC", "mac", "Modal correlation: frequency error and MAC.\n"
     "Mode tables: one row per mode: frequency, then shape values at the same sensor DOFs.", [
        ("test", "Test mode table", "file", "", None),
        ("fea", "FEA mode table", "file", "", None),
    ]),
    ("Damping", "damping", "Measure damping. decay: log decrement of a ring-down (impact or shock); "
     "psd: half-power of a response PSD peak (-3 dB); frf: multi-mode curve fit of the transmissibility "
     "between a control and a response channel (close modes).", [
        ("method", "Method", "choice", "decay", ["decay", "psd", "frf"]),
        ("file", "File (time history, or PSD for psd)", "file", "", None),
        ("channel", "decay: ring-down channel", "col", "", "file"),
        ("input", "frf: control (input) channel", "col", "", "file"),
        ("output", "frf: response channel", "col", "", "file"),
        ("fn", "Mode frequency (frf: one per mode, e.g. 98 108)", "text", "", None),
        ("band0", "Band from (Hz), optional", "float", "", None),
        ("band1", "Band to (Hz), optional", "float", "", None),
        ("fs", "Sample rate (Hz), if no Time column", "float", "", None),
    ]),
    ("UFF → CSV", "uff2csv", "Convert every dataset-58 block of a UFF file (lab export) to CSV.", [
        ("file", "UFF file (.uff / .unv)", "file", "", None),
    ]),
]


class TextRedirect:
    def __init__(self, q):
        self.q = q

    def write(self, s):
        if s:
            self.q.put(("text", s))

    def flush(self):
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"vibtools {__version__} - vibration test data and FEA correlation")
        self.geometry("1280x820")
        self.minsize(1000, 640)
        self.q = queue.Queue()
        self.busy = False
        self.vars = {}
        self.col_widgets = {}
        self.figs = []
        self.fig_index = 0
        self._style()
        self._layout()
        cli.PLOT_HOOK = lambda fig: self.q.put(("fig", fig))
        self.after(80, self._poll)
        self._log(f"vibtools {__version__}. Choose a tab, fill in the fields and press Run.\n"
                  "Example files are in the example_data folder next to the program.\n\n")

    # ------------------------------------------------------------- layout
    def _style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("vista" if sys.platform == "win32" else "clam")
        except tk.TclError:
            pass
        st.configure("Title.TLabel", font=("Segoe UI", 11, "bold"), foreground=NAVY)
        st.configure("Hint.TLabel", foreground="#52514e")
        st.configure("Run.TButton", font=("Segoe UI", 10, "bold"))

    def _layout(self):
        pan = ttk.Panedwindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=6, pady=6)
        left = ttk.Frame(pan, width=420)
        right = ttk.Frame(pan)
        pan.add(left, weight=0)
        pan.add(right, weight=1)

        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True)
        for title, cmd, hint, fields in TABS:
            self._make_tab(title, cmd, hint, fields)
        self._make_tools_tab()

        outf = ttk.Frame(left)
        outf.pack(fill="x", pady=(6, 0))
        ttk.Label(outf, text="Output folder (empty = next to the input file)").pack(anchor="w")
        row = ttk.Frame(outf); row.pack(fill="x")
        self.out_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.out_var).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="…", width=3, command=self._pick_out).pack(side="left", padx=2)
        ttk.Button(row, text="Open", command=self._open_out).pack(side="left")

        # right: plot on top, log below
        vpan = ttk.Panedwindow(right, orient="vertical")
        vpan.pack(fill="both", expand=True)
        plotf = ttk.Frame(vpan); logf = ttk.Frame(vpan)
        vpan.add(plotf, weight=3); vpan.add(logf, weight=2)

        nav = ttk.Frame(plotf); nav.pack(fill="x")
        self.fig_label = ttk.Label(nav, text="No plot yet", style="Hint.TLabel")
        ttk.Button(nav, text="◀", width=3, command=lambda: self._show_fig(self.fig_index - 1)).pack(side="left")
        ttk.Button(nav, text="▶", width=3, command=lambda: self._show_fig(self.fig_index + 1)).pack(side="left")
        self.fig_label.pack(side="left", padx=8)
        self.canvas_holder = ttk.Frame(plotf)
        self.canvas_holder.pack(fill="both", expand=True)

        bar = ttk.Frame(logf); bar.pack(fill="x")
        ttk.Label(bar, text="Results", style="Title.TLabel").pack(side="left")
        ttk.Button(bar, text="Clear", command=lambda: self.log.delete("1.0", "end")).pack(side="right")
        ttk.Button(bar, text="Copy", command=self._copy_log).pack(side="right", padx=4)
        self.log = tk.Text(logf, height=12, wrap="word", font=("Consolas", 9), relief="flat",
                           background="#fbfbfa")
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); self.log.pack(fill="both", expand=True)

        self.status = ttk.Label(self, text="Ready", anchor="w", style="Hint.TLabel")
        self.status.pack(fill="x", padx=8, pady=(0, 4))

    def _make_tab(self, title, cmd, hint, fields):
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text=title)
        ttk.Label(f, text=hint, style="Hint.TLabel", wraplength=380, justify="left").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))
        vars_ = {}
        for r, (key, label, kind, default, extra) in enumerate(fields, start=1):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3, padx=(0, 6))
            v = tk.StringVar(value=default)
            vars_[key] = v
            if kind == "file":
                e = ttk.Entry(f, textvariable=v, width=30)
                e.grid(row=r, column=1, sticky="ew")
                ttk.Button(f, text="Browse…", command=lambda v=v, c=cmd, k=key: self._browse(v, c, k)).grid(
                    row=r, column=2, padx=2)
            elif kind == "col":
                cb = ttk.Combobox(f, textvariable=v, width=28, values=["(all)"] if default == "(all)" else [])
                cb.grid(row=r, column=1, columnspan=2, sticky="ew")
                self.col_widgets.setdefault(cmd, []).append((cb, default == "(all)"))
            elif kind == "choice":
                ttk.Combobox(f, textvariable=v, values=extra, state="readonly", width=12).grid(
                    row=r, column=1, sticky="w")
            else:
                ttk.Entry(f, textvariable=v, width=30 if kind == "text" else 14).grid(
                    row=r, column=1, columnspan=2 if kind == "text" else 1, sticky="ew" if kind == "text" else "w")
        f.columnconfigure(1, weight=1)
        self.vars[cmd] = vars_
        ttk.Button(f, text="Run", style="Run.TButton", command=lambda c=cmd: self._run(c)).grid(
            row=len(fields) + 1, column=0, columnspan=3, sticky="ew", pady=(14, 0), ipady=4)

    def _make_tools_tab(self):
        f = ttk.Frame(self.nb, padding=10)
        self.nb.add(f, text="Checks")
        items = [
            ("Self-test", "selftest", "Check core numbers against published values (Irvine SRS, ISO 5349-1 W_h, "
                                      "closed-form PSD integrals)."),
            ("Benchmark", "benchmark", "Reference values for an FEA benchmark: "
                                                    "cantilever with tip mass on a shaker."),
            ("Demo", "demo", "Synthetic test vs FEA example: transmissibility, PSD, SRS and fatigue."),
        ]
        for r, (label, cmd, hint) in enumerate(items):
            ttk.Button(f, text=label, style="Run.TButton", command=lambda c=cmd: self._run(c)).grid(
                row=2 * r, column=0, sticky="ew", pady=(10 if r else 0, 2), ipady=3)
            ttk.Label(f, text=hint, style="Hint.TLabel", wraplength=360, justify="left").grid(
                row=2 * r + 1, column=0, sticky="w")
        f.columnconfigure(0, weight=1)
        ttk.Label(f, text=f"\nvibtools {__version__}\ngithub.com/hamidattaran/vibtools"
                          "\nThe W_h filter is not a certified instrument.",
                  style="Hint.TLabel", justify="left").grid(row=10, column=0, sticky="w", pady=(20, 0))

    # ------------------------------------------------------------- helpers
    def _browse(self, var, cmd, key):
        start = os.path.dirname(var.get()) if var.get() else self._example_dir()
        p = filedialog.askopenfilename(initialdir=start, filetypes=[
            ("Data files", "*.csv *.txt *.dat *.tsv *.uff *.unv"), ("All files", "*.*")])
        if not p:
            return
        var.set(p)
        if key == "file" and cmd in self.col_widgets:
            self._fill_columns(cmd, p)

    def _example_dir(self):
        base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
        d = os.path.join(base, "example_data")
        return d if os.path.isdir(d) else base

    def _fill_columns(self, cmd, path):
        try:
            data, units = cli._load(path)
        except Exception as exc:                     # noqa: BLE001
            messagebox.showerror("vibtools", f"Could not read {path}:\n{exc}")
            return
        cols = [k for k in data if k.lower() not in ("time", "t", "time_s", "x")]
        for k, (cb, allow_all) in enumerate(self.col_widgets[cmd]):
            cb["values"] = (["(all)"] if allow_all else []) + cols
            cur = cb.get()
            if cur not in cb["values"]:
                cb.set("(all)" if allow_all else (cols[min(k, len(cols) - 1)] if cols else ""))
        names = [f"{c} [{units[c]}]" if units.get(c) else c for c in data]
        self._log(f"{os.path.basename(path)}: {len(next(iter(data.values())))} rows; columns: "
                  f"{', '.join(names)}\n")

    def _pick_out(self):
        d = filedialog.askdirectory()
        if d:
            self.out_var.set(d)

    def _open_out(self):
        d = self.out_var.get() or self._last_out_dir()
        if d and os.path.isdir(d):
            os.startfile(d) if sys.platform == "win32" else None   # noqa: S606
        else:
            messagebox.showinfo("vibtools", "No output folder yet: run a tool first or choose a folder.")

    def _last_out_dir(self):
        return getattr(self, "_last_dir", "")

    def _copy_log(self):
        self.clipboard_clear(); self.clipboard_append(self.log.get("1.0", "end"))

    def _log(self, s):
        self.log.insert("end", s); self.log.see("end")

    # ------------------------------------------------------------- run
    def _num(self, v, kind=float):
        v = v.strip()
        return kind(float(v)) if v else None

    def _namespace(self, cmd):
        v = {k: var.get().strip() for k, var in self.vars.get(cmd, {}).items()}
        a = argparse.Namespace(out=self.out_var.get().strip() or None)
        try:
            if cmd == "psd":
                a.file, a.fs, a.df = v["file"], self._num(v["fs"]), self._num(v["df"])
                a.channel = None if v["channel"] in ("", "(all)") else v["channel"]
                a.band, a.units = (self._num(v["band0"]), self._num(v["band1"])), v["units"]
            elif cmd == "frf":
                a.file, a.input, a.output, a.fs = v["file"], v["input"] or None, v["output"] or None, self._num(v["fs"])
                a.df, a.band = self._num(v["df"]), (self._num(v["band0"]), self._num(v["band1"]))
                a.min_coherence = self._num(v["min_coherence"])
            elif cmd == "srs":
                a.file, a.channel, a.fs, a.units = v["file"], v["channel"] or None, self._num(v["fs"]), v["units"]
                a.q, a.fmin, a.fmax = self._num(v["q"]), self._num(v["fmin"]), self._num(v["fmax"])
                a.n_per_oct = self._num(v["n_per_oct"], int)
            elif cmd == "profile":
                a.breakpoints, a.file = v["breakpoints"] or None, v["file"] or None
                b0, b1 = self._num(v["band0"]), self._num(v["band1"])
                a.band = (b0, b1) if b0 is not None and b1 is not None else None
                a.fn, a.q = self._num(v["fn"]), self._num(v["q"]) or 10.0
            elif cmd == "fatigue":
                a.file, a.time = v["file"], self._num(v["time"])
                a.sn_points = [f"{v['s1']},{v['n1']}", f"{v['s2']},{v['n2']}"]
                a.sn_c = a.sn_m = None
            elif cmd == "fds":
                a.breakpoints, a.file = v["breakpoints"] or None, v["file"] or None
                a.time, a.q, a.b = self._num(v["time"]), self._num(v["q"]), self._num(v["b"])
                a.fmin, a.fmax = self._num(v["fmin"]), self._num(v["fmax"])
                a.compare, a.compare_time = v["compare"] or None, self._num(v["compare_time"])
            elif cmd == "hav":
                a.file, a.fs, a.units, a.hours = v["file"], self._num(v["fs"]), v["units"], self._num(v["hours"])
                a.x, a.y, a.z = v["x"] or None, v["y"] or None, v["z"] or None
            elif cmd == "mac":
                a.test, a.fea = v["test"], v["fea"]
            elif cmd == "uff2csv":
                a.file = v["file"]
            elif cmd == "damping":
                a.method, a.file, a.fs, a.df = v["method"], v["file"], self._num(v["fs"]), 0.25
                a.channel, a.input, a.output = v["channel"] or None, v["input"] or None, v["output"] or None
                a.fn = [float(x) for x in v["fn"].replace(",", " ").split()] or None
                b0, b1 = self._num(v["band0"]), self._num(v["band1"])
                a.band = [b0, b1] if b0 is not None and b1 is not None else None
                if a.method == "frf" and not a.fn:
                    raise ValueError("frf method: give one starting frequency per mode")
        except ValueError as exc:
            raise ValueError(f"check the numeric fields ({exc})") from None
        need = {"psd": ["file"], "frf": ["file"], "srs": ["file"], "fatigue": ["file", "time"],
                "fds": ["time"], "hav": ["file"], "mac": ["test", "fea"], "uff2csv": ["file"],
                "damping": ["file"]}.get(cmd, [])
        for k in need:
            if not getattr(a, k, None):
                raise ValueError(f"please fill in: {k}")
        if cmd in ("profile", "fds") and not (a.breakpoints or a.file):
            raise ValueError("give breakpoints or choose an ASD file")
        return a

    def _run(self, cmd):
        if self.busy:
            return
        try:
            args = self._namespace(cmd)
        except ValueError as exc:
            messagebox.showwarning("vibtools", str(exc)); return
        src = getattr(args, "file", None) or getattr(args, "test", None)
        self._last_dir = args.out or (os.path.dirname(os.path.abspath(src)) if src else os.getcwd())
        if cmd in ("demo", "benchmark"):
            args.out = args.out or self._default_check_dir()
            self._last_dir = args.out
        self.busy = True
        self.status.configure(text=f"Running {cmd} …")
        self.config(cursor="watch")
        self._log(f"\n=== {cmd} ===\n")
        threading.Thread(target=self._worker, args=(cmd, args), daemon=True).start()

    def _default_check_dir(self):
        base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.getcwd()
        d = os.path.join(base, "results")
        os.makedirs(d, exist_ok=True)
        return d

    def _worker(self, cmd, args):
        red = TextRedirect(self.q)
        try:
            with contextlib.redirect_stdout(red), contextlib.redirect_stderr(red):
                fn = getattr(cli, f"cmd_{cmd}")
                fn(args)
                if cmd in ("demo", "benchmark"):
                    png = os.path.join(args.out, "demo_output.png" if cmd == "demo" else "benchmark_cantilever.png")
                    if os.path.exists(png):
                        self.q.put(("image", png))
            self.q.put(("done", cmd))
        except SystemExit as exc:
            self.q.put(("error", str(exc)))
        except Exception:                            # noqa: BLE001
            self.q.put(("error", traceback.format_exc(limit=3)))

    def _poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "text":
                    self._log(payload)
                elif kind == "fig":
                    self.figs.append(payload); self._show_fig(len(self.figs) - 1)
                elif kind == "image":
                    self._image_fig(payload)
                elif kind in ("done", "error"):
                    if kind == "error":
                        self._log(f"ERROR: {payload}\n")
                        messagebox.showerror("vibtools", payload[-600:])
                    self.busy = False
                    self.config(cursor="")
                    self.status.configure(text="Ready" if kind == "done" else "Finished with an error")
        except queue.Empty:
            pass
        self.after(80, self._poll)

    def _image_fig(self, png):
        import matplotlib.image as mpimg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(7, 4.5))
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(mpimg.imread(png)); ax.axis("off")
        self.figs.append(fig); self._show_fig(len(self.figs) - 1)

    def _show_fig(self, i):
        if not self.figs:
            return
        i = max(0, min(i, len(self.figs) - 1))
        self.fig_index = i
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        for w in self.canvas_holder.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(self.figs[i], master=self.canvas_holder)
        tb = NavigationToolbar2Tk(canvas, self.canvas_holder, pack_toolbar=False)
        tb.update(); tb.pack(side="bottom", fill="x")
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        self.fig_label.configure(text=f"Plot {i + 1} of {len(self.figs)}  (toolbar: zoom, pan, save)")


def _autotest(app, spec):
    """Packaging check: VIBTOOLS_GUI_AUTOTEST="<example_dir>|<log_file>" runs three tabs and exits."""
    ex, logf = spec.split("|")
    steps = [("frf", {"file": "shaker_random_run.csv", "input": "control", "output": "tip"}, 1),
             ("hav", {"file": "handle_3axis.csv", "hours": "2"}, 6),
             ("damping", {"method": "frf", "file": "shaker_close_modes.csv", "input": "control",
                          "output": "bracket", "fn": "98 108", "band0": "80", "band1": "130"}, 9),
             ("benchmark", {}, 10)]
    app.out_var.set(os.path.join(os.path.dirname(logf), "autotest_out"))

    def step(i=0):
        if app.busy:
            app.after(200, lambda: step(i)); return
        if i == len(steps):
            try:
                from PIL import ImageGrab
                app.update(); x, y = app.winfo_rootx(), app.winfo_rooty()
                ImageGrab.grab(bbox=(x, y, x + app.winfo_width(), y + app.winfo_height())).save(
                    os.path.join(os.path.dirname(logf), "autotest_window.png"))
            except Exception:                        # noqa: BLE001
                pass
            with open(logf, "w", encoding="utf-8") as fh:
                fh.write(app.log.get("1.0", "end")); fh.write(f"\nPLOTS {len(app.figs)}\n")
            app.after(300, app.destroy); return
        cmd, fields, tab = steps[i]
        app.nb.select(tab)
        for k, v in fields.items():
            app.vars[cmd][k].set(os.path.join(ex, v) if k == "file" else v)
            if k == "file" and cmd in app.col_widgets:
                app._fill_columns(cmd, os.path.join(ex, v))
        for k in ("input", "output", "method", "fn", "band0", "band1"):
            if k in fields:
                app.vars[cmd][k].set(fields[k])
        app._run(cmd)
        app.after(500, lambda: step(i + 1))
    app.after(1500, step)


def main():
    app = App()
    spec = os.environ.get("VIBTOOLS_GUI_AUTOTEST")
    if spec:
        _autotest(app, spec)
    app.mainloop()


if __name__ == "__main__":
    main()
