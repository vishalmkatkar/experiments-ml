"""Settings for the fleet anomaly series. Everything that changes the generated fleet
or a run's behaviour lives here and is logged to MLflow with the run."""

import os
from pathlib import Path

SEED = int(os.getenv("FA_SEED", "42"))

# Fleet size. The full fleet is ~200 machines x 8,760 hours ≈ 1.75M telemetry rows;
# --fast shrinks it for quick checks.
N_MACHINES = int(os.getenv("FA_MACHINES", "200"))
N_DAYS = int(os.getenv("FA_DAYS", "365"))
FAST_MACHINES = 30
FAST_DAYS = 120
START = "2026-01-01T00:00:00"

SENSORS = ["vibration", "temperature", "pressure", "motor_current", "flow", "speed"]

PACKAGE_DIR = Path(__file__).parent
REPO_DIR = PACKAGE_DIR.parents[2]
DATA_DIR = Path(os.getenv("FA_DATA_DIR", PACKAGE_DIR / "data" / "generated"))
TRUTH_DIR = DATA_DIR / "truth"
SAMPLES_DIR = PACKAGE_DIR / "samples"
OUT_DIR = PACKAGE_DIR / "out"

# Spark runs in local mode on an M1 with 16 GB. Spark 4.x supports Java 17 and 21;
# the repo points at a Homebrew JDK 21 without changing the system default.
JAVA_HOME = os.getenv("FA_JAVA_HOME", "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home")
SPARK_MASTER = os.getenv("FA_SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.getenv("FA_SPARK_DRIVER_MEMORY", "6g")
SPARK_SHUFFLE_PARTITIONS = int(os.getenv("FA_SPARK_SHUFFLE_PARTITIONS", "8"))

# MLflow tracks to a local SQLite database at the repo root (both paths are gitignored).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{REPO_DIR / 'mlflow.db'}")
MLFLOW_ARTIFACT_DIR = REPO_DIR / "mlartifacts"
MLFLOW_EXPERIMENT = "fleet-anomaly"
