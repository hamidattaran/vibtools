"""Run the bundled example scripts (works from source and from the frozen executable)."""
import os
import runpy
import sys


def _examples_dir():
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "examples")          # noqa: SLF001 (PyInstaller)
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")


def _run(script, out_dir, argv):
    path = os.path.join(_examples_dir(), script)
    if not os.path.exists(path):
        sys.exit(f"example not found: {path}")
    os.makedirs(out_dir, exist_ok=True)
    old_env, old_argv = os.environ.get("VIBTOOLS_OUT"), sys.argv
    os.environ["VIBTOOLS_OUT"] = os.path.abspath(out_dir)
    sys.argv = [path] + argv
    try:
        runpy.run_path(path, run_name="__main__")
    finally:
        sys.argv = old_argv
        if old_env is None:
            os.environ.pop("VIBTOOLS_OUT", None)
        else:
            os.environ["VIBTOOLS_OUT"] = old_env


def run_demo(out_dir):
    _run("demo_correlation.py", out_dir, [])


def run_benchmark(out_dir):
    _run("benchmark_cantilever.py", out_dir, ["--plot"])
