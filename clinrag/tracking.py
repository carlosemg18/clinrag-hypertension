"""MLflow setup. Local file-backed tracking store; no server required.

Usage:
    from clinrag.tracking import init_tracking, start_run

    init_tracking()
    with start_run("eval-claude"):
        mlflow.log_metric("faithfulness", 0.91)

Launch the UI to browse runs:
    mlflow ui --backend-store-uri file:./mlruns
"""

from __future__ import annotations

from contextlib import contextmanager

import mlflow

from clinrag.config import SETTINGS


def init_tracking() -> None:
    """Point MLflow at the local store and ensure the experiment exists."""
    mlflow.set_tracking_uri(SETTINGS.mlflow_tracking_uri)
    mlflow.set_experiment(SETTINGS.mlflow_experiment)


@contextmanager
def start_run(run_name: str, **tags: str):
    """Start an MLflow run under the project experiment."""
    init_tracking()
    with mlflow.start_run(run_name=run_name, tags=tags or None) as run:
        yield run


if __name__ == "__main__":
    # Smoke test: write a trivial run so `mlflow ui` has something to show.
    init_tracking()
    with start_run("setup-smoke-test", phase="day1-2"):
        mlflow.log_param("status", "tracking-initialized")
    print(f"MLflow ready. Tracking URI: {SETTINGS.mlflow_tracking_uri}")
    print("View with: mlflow ui --backend-store-uri", SETTINGS.mlflow_tracking_uri)
