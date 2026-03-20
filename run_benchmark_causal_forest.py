"""
Run the causal forest GATE vs DML benchmark on Databricks serverless compute.

Install notes (for this cluster, Databricks serverless 2026-03):
- numpy must be <2 (Databricks pyarrow compiled for numpy 1.x)
- scikit-learn must be <1.6 (sklearn 1.8+ in venv imports Databricks pyarrow and breaks)
- econml latest fails wheel build; econml==0.15.1 works
- sparse must be installed BEFORE econml
"""
import os
import sys
import time
import base64
import pathlib
import re

env_path = pathlib.Path.home() / ".config/burning-cost/databricks.env"
for line in env_path.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service import jobs

w = WorkspaceClient()

WORKSPACE_PATH = "/Workspace/tmp/insurance-causal-cf-benchmark"
REPO_ROOT = pathlib.Path("/home/ralph/repos/insurance-causal")


def upload_file(local_path: pathlib.Path, ws_path: str) -> None:
    content = base64.b64encode(local_path.read_bytes()).decode()
    parent = "/".join(ws_path.split("/")[:-1])
    try:
        w.workspace.mkdirs(path=parent)
    except Exception:
        pass
    w.workspace.import_(
        path=ws_path,
        content=content,
        format=ImportFormat.AUTO,
        overwrite=True,
    )


patterns = ["src/**/*.py", "benchmarks/**/*.py", "pyproject.toml"]
uploaded = 0
for pattern in patterns:
    for fpath in REPO_ROOT.glob(pattern):
        if "__pycache__" in str(fpath):
            continue
        relative = fpath.relative_to(REPO_ROOT)
        ws_path = f"{WORKSPACE_PATH}/{relative}"
        try:
            upload_file(fpath, ws_path)
            uploaded += 1
        except Exception as e:
            print(f"  FAIL {relative}: {e}")

print(f"Uploaded {uploaded} files")

notebook_content = """# Databricks notebook source
import subprocess, sys, os, shutil

install_script = \"\"\"
import subprocess, sys

def pip(*pkgs):
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet'] + list(pkgs),
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print('FAILED pip install', pkgs[:3])
        print(r.stderr[-400:])
    return r.returncode == 0

pip('numpy>=1.25,<2')
pip('scipy>=1.11', 'scikit-learn>=1.3,<1.6', 'doubleml', 'catboost', 'polars', 'joblib')
pip('sparse')
econml_ok = pip('econml==0.15.1')
print('econml_ok:', econml_ok)
import numpy as np; print(f'numpy {np.__version__}')
try:
    from econml.dml import CausalForestDML; print('CausalForestDML: ok')
except Exception as e:
    print(f'econml import error: {e}')
\"\"\"

install_r = subprocess.run([sys.executable, '-c', install_script], capture_output=True, text=True)
install_out = install_r.stdout[-2000:]
install_err = install_r.stderr[-200:]

src_ws = '/Workspace/tmp/insurance-causal-cf-benchmark'
dst_tmp = '/tmp/insurance-causal-cf-benchmark'
if os.path.exists(dst_tmp):
    shutil.rmtree(dst_tmp)
shutil.copytree(src_ws, dst_tmp, ignore=shutil.ignore_patterns('*.ipynb'))

env = {**os.environ, 'PYTHONPATH': f'{dst_tmp}/src', 'PYTHONWARNINGS': 'ignore'}
result = subprocess.run(
    [sys.executable, '-W', 'ignore', 'benchmarks/benchmark_causal_forest.py'],
    capture_output=True, text=True, cwd=dst_tmp, env=env, timeout=1800,
)
rc = result.returncode
full_output = (
    '=== INSTALL ===\\n' + install_out +
    '\\n=== BENCHMARK STDOUT ===\\n' + result.stdout +
    '\\n=== STDERR (last 1500) ===\\n' + result.stderr[-1500:]
)
dbutils.notebook.exit(f'EXIT_CODE={rc}\\n{full_output[-18000:]}')
"""

nb_path = f"{WORKSPACE_PATH}/run_cf_benchmark_nb"
nb_b64 = base64.b64encode(notebook_content.encode()).decode()

try:
    w.workspace.mkdirs(path=WORKSPACE_PATH)
except Exception:
    pass

w.workspace.import_(
    path=nb_path,
    content=nb_b64,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print(f"Notebook: {nb_path}")

print("Submitting serverless job...")
run_waiter = w.jobs.submit(
    run_name="insurance-causal-cf-benchmark-v23",
    tasks=[
        jobs.SubmitTask(
            task_key="cf_benchmark",
            notebook_task=jobs.NotebookTask(notebook_path=nb_path),
        )
    ],
)

run_id = run_waiter.run_id
print(f"Run ID: {run_id}")
print(f"URL: {os.environ['DATABRICKS_HOST']}#job/run/{run_id}")

while True:
    run_info = w.jobs.get_run(run_id=run_id)
    state = run_info.state
    lc = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    print(f"  [{lc}]", flush=True)
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        rc_val = state.result_state.value if state.result_state else "UNKNOWN"
        print(f"Result: {rc_val}")
        for task in (run_info.tasks or []):
            try:
                out = w.jobs.get_run_output(run_id=task.run_id)
                if out.notebook_output and out.notebook_output.result:
                    clean = re.sub(r'\x1b\[[0-9;]*m', '', out.notebook_output.result)
                    print("\n--- Output ---")
                    print(clean)
                if out.error:
                    print("Error:", out.error)
                if out.error_trace:
                    trace = re.sub(r'\x1b\[[0-9;]*m', '', out.error_trace)
                    print("Trace:", trace[-3000:])
            except Exception as e:
                print(f"Could not get output: {e}")
        sys.exit(0 if rc_val == "SUCCESS" else 1)
    time.sleep(30)
