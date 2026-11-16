"""MLflow, tracking to a local SQLite database at the repository root.

Every part logs its runs to one experiment, "fleet-anomaly", so generation,
clustering and detector runs sit side by side and can be compared.
"""

import contextlib
import json
import logging
import os
from pathlib import Path

from . import config

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@contextlib.contextmanager
def start_run(name: str, tags: dict | None = None):
    import mlflow

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("alembic").setLevel(logging.WARNING)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        config.MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(config.MLFLOW_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=name, tags=tags or {}) as run:
        yield run


def log_dict_flat(prefix: str, d: dict) -> None:
    """Log the numeric leaves of a nested dict as metrics, e.g. failures_by_mode.seal_leak."""
    import mlflow

    metrics = {}

    def walk(p, v):
        if isinstance(v, dict):
            for k, x in v.items():
                walk(f"{p}.{k}" if p else str(k), x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            metrics[p.replace("/", ".")] = float(v)

    walk(prefix, d)
    mlflow.log_metrics(metrics)


def log_json(obj: dict, artifact_file: str) -> None:
    import mlflow

    mlflow.log_dict(json.loads(json.dumps(obj, default=str)), artifact_file)


def log_file(path: Path) -> None:
    import mlflow

    mlflow.log_artifact(str(path))
