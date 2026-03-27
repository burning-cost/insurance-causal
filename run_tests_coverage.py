"""
Submit the insurance-causal test suite to Databricks with coverage reporting.

Usage: python run_tests_coverage.py
"""

import os
import base64
import time
import pathlib

# Load credentials
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k] = v

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
from databricks.sdk.service import workspace as ws_service

w = WorkspaceClient()

WORKSPACE_PATH = "/Workspace/insurance-causal-coverage"
NOTEBOOK_PATH = f"{WORKSPACE_PATH}/run_pytest_coverage"

notebook_source = '''# Databricks notebook source
# MAGIC %pip install "insurance-causal[all,dev]" pytest-cov --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import subprocess, sys

# First run the full suite to check pass/fail
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-x", "-q",
     "--tb=short",
     "--cov=insurance_causal",
     "--cov-report=term-missing",
     "--cov-report=term:skip-covered",
     "/Workspace/insurance-causal-coverage/tests"],
    capture_output=True, text=True
)

print("=== STDOUT ===")
print(result.stdout[-12000:])
print("=== STDERR ===")
print(result.stderr[-4000:])

if result.returncode != 0:
    raise SystemExit(f"pytest failed with return code {result.returncode}")

print("All tests passed.")
'''

repo_root = pathlib.Path("/home/ralph/repos/insurance-causal")


def upload_file(local_path: pathlib.Path, workspace_path: str):
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    try:
        w.workspace.import_(
            path=workspace_path,
            content=encoded,
            format=ws_service.ImportFormat.AUTO,
            overwrite=True,
        )
    except Exception as e:
        print(f"  Warning uploading {workspace_path}: {e}")


# Create base directories
for d in [WORKSPACE_PATH, f"{WORKSPACE_PATH}/tests"]:
    try:
        w.workspace.mkdirs(d)
    except Exception:
        pass

# Upload the notebook
print("Uploading notebook...")
notebook_encoded = base64.b64encode(notebook_source.encode()).decode()
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=notebook_encoded,
    format=ws_service.ImportFormat.SOURCE,
    language=ws_service.Language.PYTHON,
    overwrite=True,
)
print(f"  Notebook: {NOTEBOOK_PATH}")

# Upload test files
print("Uploading test files...")
tests_dir = repo_root / "tests"
for test_file in sorted(tests_dir.rglob("*.py")):
    rel = test_file.relative_to(repo_root)
    ws_path = f"{WORKSPACE_PATH}/{rel}"
    parent = str(pathlib.PurePosixPath(ws_path).parent)
    try:
        w.workspace.mkdirs(parent)
    except Exception:
        pass
    upload_file(test_file, ws_path)
    print(f"  {rel}")

# Submit using serverless compute
print("\nSubmitting job run on serverless compute...")
run_waiter = w.jobs.submit(
    run_name="insurance-causal pytest coverage",
    tasks=[
        jobs.SubmitTask(
            task_key="pytest-coverage",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
            ),
            environment_key="default",
        )
    ],
    environments=[
        jobs.JobEnvironment(
            environment_key="default",
            spec=compute.Environment(
                client="2",
            ),
        )
    ],
)

run_id = run_waiter.run_id
print(f"  Run ID: {run_id}")
print(f"  URL: {os.environ['DATABRICKS_HOST']}#job/runs/{run_id}")

# Poll for completion
print("Waiting for run to complete...")
while True:
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = run_state.state.life_cycle_state
    result_state = run_state.state.result_state
    print(f"  State: {life_cycle} / {result_state}")
    if life_cycle in (
        jobs.RunLifeCycleState.TERMINATED,
        jobs.RunLifeCycleState.SKIPPED,
        jobs.RunLifeCycleState.INTERNAL_ERROR,
    ):
        break
    time.sleep(30)

# Get task output
for task in (run_state.tasks or []):
    try:
        output = w.jobs.get_run_output(run_id=task.run_id)
        if output.notebook_output and output.notebook_output.result:
            print("\n=== Notebook output ===")
            print(output.notebook_output.result)
        if output.error:
            print(f"\n=== Error ===\n{output.error}")
        if output.error_trace:
            print(f"\n=== Error trace (last 4000 chars) ===")
            print(output.error_trace[-4000:])
    except Exception as e:
        print(f"  Could not get output for task {task.task_key}: {e}")

final = run_state.state.result_state
print(f"\nFinal result: {final}")
if final != jobs.RunResultState.SUCCESS:
    raise SystemExit(f"Job failed: {final}")

print("SUCCESS - all tests passed on Databricks.")
