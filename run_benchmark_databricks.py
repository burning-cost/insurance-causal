"""
Run insurance-causal benchmark on Databricks serverless compute.

Uploads src/, benchmarks/, and pyproject.toml to the workspace, runs the
benchmark as a notebook job, and prints the output.
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

WORKSPACE_PATH = "/Workspace/tmp/insurance-causal-benchmark"
REPO_ROOT = pathlib.Path("/home/ralph/burning-cost/repos/insurance-causal")


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

notebook_content = r"""# Databricks notebook source
# MAGIC %pip install "numpy<2" "scikit-learn>=1.4,<1.6" "doubleml>=0.9" "catboost>=1.2" "polars>=0.20" "scipy>=1.11" "joblib>=1.3" --quiet

# COMMAND ----------
import subprocess, sys, os, shutil, warnings

src_ws = "/Workspace/tmp/insurance-causal-benchmark"
dst_tmp = "/tmp/insurance-causal-benchmark"
if os.path.exists(dst_tmp):
    shutil.rmtree(dst_tmp)
shutil.copytree(src_ws, dst_tmp, ignore=shutil.ignore_patterns("*.ipynb"))
print("Copied to", dst_tmp)

env = {**os.environ, "PYTHONPATH": f"{dst_tmp}/src", "PYTHONWARNINGS": "ignore"}

result = subprocess.run(
    [sys.executable, "-W", "ignore", "benchmarks/run_benchmark.py"],
    capture_output=True, text=True,
    cwd=dst_tmp,
    env=env
)
# Only show stdout — stderr is full of convergence warnings from bootstrap GLM
stdout = result.stdout
stderr = result.stderr
rc = result.returncode
# Trim stderr to last 500 chars to only show real errors
stderr_trim = ""
if rc != 0 and stderr:
    stderr_trim = "\nSTDERR (last 500 chars):\n" + stderr[-500:]
dbutils.notebook.exit(f"EXIT_CODE={rc}\n{stdout[-8000:]}{stderr_trim}")
"""

nb_path = f"{WORKSPACE_PATH}/run_benchmark_nb"
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
    run_name="insurance-causal-benchmark-v7",
    tasks=[
        jobs.SubmitTask(
            task_key="benchmark",
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
                    print("\n--- Benchmark Output ---")
                    print(clean)
                if out.error:
                    print("Error:", out.error)
                if out.error_trace:
                    trace = re.sub(r'\x1b\[[0-9;]*m', '', out.error_trace)
                    print("Trace:", trace[-3000:])
            except Exception as e:
                print(f"Could not get output: {e}")
        sys.exit(0 if rc_val == "SUCCESS" else 1)
    time.sleep(20)
