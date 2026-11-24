"""File helpers: ANSYS Workbench text/CSV exports and UFF dataset 58 (test data).

read_table      : any delimited text table with a header row, for example a Workbench
                  chart or tabular-data export ("Frequency [Hz]  Amplitude [mm]").
read_uff58      : universal file format dataset 58 (function at nodal DOF): FRFs,
                  PSDs, time histories exported by most test-lab software (ASCII only).
write_uff58     : write a real or complex function as dataset 58 (ASCII), for
                  round-trip checks or for sending FEA results to the lab software.
"""
from __future__ import annotations

import re

import numpy as np

_UNIT_BRACKET = re.compile(r"^\s*([^\[]*?)\s*\[(.*)\]\s*$")     # Name [unit], unit may contain ()
_UNIT_PAREN = re.compile(r"^\s*([^(]*?)\s*\((.*)\)\s*$")        # Name (unit)


def _split(line, delim):
    if delim is None:
        return line.split()
    return [c.strip() for c in line.split(delim)]


def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_table(path, delimiter=None, encoding="utf-8-sig"):
    """Read a delimited text table with one header row.

    Detects tab, comma or semicolon delimiters (whitespace otherwise), skips any
    preamble lines before the header, and splits units in brackets from the names:
    "Frequency [Hz]" becomes key "Frequency" with unit "Hz".

    Returns (data, units): data maps column name -> float array; units maps
    column name -> unit string ("" if none). Non-numeric cells become NaN.
    """
    with open(path, encoding=encoding, errors="replace") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh if ln.strip()]
    if delimiter is None:
        sample = "\n".join(lines[:20])
        for d in ("\t", ";", ","):
            if sample.count(d) >= 2:
                delimiter = d
                break
    # header = last non-numeric line before the first numeric data row
    first_data = next(i for i, ln in enumerate(lines)
                      if all(_is_number(c) for c in _split(ln, delimiter) if c))
    header = _split(lines[first_data - 1], delimiter) if first_data > 0 else \
        [f"col{k}" for k in range(len(_split(lines[0], delimiter)))]
    names, units = [], {}
    for h in header:
        m = _UNIT_BRACKET.match(h) or _UNIT_PAREN.match(h)
        name, unit = (m.group(1), m.group(2)) if m else (h.strip(), "")
        base, k = name or "col", 1
        while name in names or not name:
            name = f"{base}_{k}"; k += 1
        names.append(name); units[name] = unit
    rows = []
    for ln in lines[first_data:]:
        cells = _split(ln, delimiter)
        rows.append([float(c) if _is_number(c) else np.nan for c in cells[:len(names)]]
                    + [np.nan] * max(0, len(names) - len(cells)))
    arr = np.asarray(rows, float)
    return {n: arr[:, k] for k, n in enumerate(names)}, units


# --------------------------------------------------------------------------
# UFF dataset 58
# --------------------------------------------------------------------------
_FUNC_TYPES = {0: "general", 1: "time response", 2: "auto spectrum", 3: "cross spectrum",
               4: "frequency response function", 5: "transmissibility", 6: "coherence",
               9: "power spectral density", 12: "spectrum"}


def _fixed(line, width):
    line = line.rstrip()
    return [float(line[i:i + width]) for i in range(0, len(line), width) if line[i:i + width].strip()]


def read_uff58(path):
    """Read all dataset-58 blocks of an ASCII UFF/UNV file.

    Returns a list of dicts with keys: id (5 text lines), func_type, func_name,
    node/direction (response and reference), x (abscissa), y (real or complex),
    x_label, x_unit, y_label, y_unit (from the axis records).
    """
    with open(path, encoding="latin-1") as fh:
        lines = [ln.rstrip("\r\n") for ln in fh]
    out, i = [], 0
    while i < len(lines):
        if lines[i].strip() == "-1" and i + 1 < len(lines) and lines[i + 1].strip() in ("58", "58b"):
            if lines[i + 1].strip() == "58b":
                raise NotImplementedError("binary dataset 58b is not supported; export ASCII")
            h = lines[i + 2:i + 13]
            r6, r7 = h[5].split(), h[6].split()
            ftype = int(r6[0])
            ordtype, npts, even = int(r7[0]), int(r7[1]), int(r7[2])
            xmin, dx = float(r7[3]), float(r7[4])
            cplx = ordtype in (5, 6)
            width = 20 if ordtype in (4, 6) else 13
            vals, j = [], i + 13
            while lines[j].strip() != "-1":
                vals.extend(_fixed(lines[j], width)); j += 1
            vals = np.asarray(vals)
            if even:
                x = xmin + dx * np.arange(npts)
                y = vals[0::2] + 1j * vals[1::2] if cplx else vals
            else:
                step = 3 if cplx else 2
                x = vals[0::step]
                y = vals[1::step] + 1j * vals[2::step] if cplx else vals[1::step]
            def axis(rec):
                return rec[26:46].strip(), rec[47:67].strip()
            out.append(dict(id=[ln.strip() for ln in h[0:5]], func_type=ftype,
                            func_name=_FUNC_TYPES.get(ftype, str(ftype)),
                            response_node=int(r6[5]) if len(r6) > 5 else None,
                            response_dir=int(r6[6]) if len(r6) > 6 else None,
                            reference_node=int(r6[8]) if len(r6) > 8 else None,
                            reference_dir=int(r6[9]) if len(r6) > 9 else None,
                            x=np.asarray(x[:npts]), y=np.asarray(y[:npts]),
                            x_label=axis(h[7])[0], x_unit=axis(h[7])[1],
                            y_label=axis(h[8])[0], y_unit=axis(h[8])[1]))
            i = j + 1
        else:
            i += 1
    return out


def write_uff58(path, x, y, func_type=4, name="vibtools", response=(1, 3), reference=(0, 0),
                x_label="Frequency", y_label="Response", x_unit="Hz", y_unit="", append=False):
    """Write one evenly spaced real or complex function as an ASCII dataset 58 block."""
    x = np.asarray(x, float)
    y = np.asarray(y)
    cplx = np.iscomplexobj(y)
    ordtype = 6 if cplx else 4
    vals = np.column_stack([y.real, y.imag]).ravel() if cplx else y.astype(float)
    dx = float(x[1] - x[0]) if x.size > 1 else 0.0
    rec = ["    -1", "    58",
           f"{name:<80}"[:80], "NONE".ljust(80), "NONE".ljust(80), "NONE".ljust(80), "NONE".ljust(80),
           f"{func_type:5d}{0:10d}{0:5d}{0:10d} {'NONE':<10}{response[0]:10d}{response[1]:4d} "
           f"{'NONE':<10}{reference[0]:10d}{reference[1]:4d}",
           f"{ordtype:10d}{x.size:10d}{1:10d}{x[0]:13.5E}{dx:13.5E}{0.0:13.5E}",
           f"{18:10d}{0:5d}{0:5d}{0:5d} {x_label[:20]:<20} {x_unit[:20]:<20}",
           f"{0:10d}{0:5d}{0:5d}{0:5d} {y_label[:20]:<20} {y_unit[:20]:<20}",
           f"{0:10d}{0:5d}{0:5d}{0:5d} {'NONE':<20} {'NONE':<20}",
           f"{0:10d}{0:5d}{0:5d}{0:5d} {'NONE':<20} {'NONE':<20}"]
    for k in range(0, vals.size, 4):
        rec.append("".join(f"{v:20.11E}" for v in vals[k:k + 4]))
    rec.append("    -1")
    with open(path, "a" if append else "w", encoding="latin-1") as fh:
        fh.write("\n".join(rec) + "\n")
