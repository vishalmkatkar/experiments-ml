"""One local SparkSession for the whole series.

Spark 4.x runs on Java 17 or 21. JAVA_HOME is set for this process only, from
config.JAVA_HOME (a Homebrew JDK 21 by default), so the system Java stays as it is.
"""

import os
import subprocess
from pathlib import Path

from . import config

_spark = None


def java_version(java_home: str) -> str:
    java = Path(java_home) / "bin" / "java"
    if not java.exists():
        raise RuntimeError(f"No Java at {java}. Install a JDK 17 or 21 (brew install openjdk@21) "
                           f"or set FA_JAVA_HOME.")
    out = subprocess.run([str(java), "-version"], capture_output=True, text=True).stderr
    return out.splitlines()[0] if out else "unknown"


def get_spark():
    global _spark
    if _spark is not None:
        return _spark
    version = java_version(config.JAVA_HOME)
    if not any(f'"{v}' in version for v in ("17", "21")):
        raise RuntimeError(f"Spark 4.x needs Java 17 or 21, found {version} at {config.JAVA_HOME}")
    os.environ["JAVA_HOME"] = config.JAVA_HOME

    from pyspark.sql import SparkSession

    _spark = (SparkSession.builder
              .master(config.SPARK_MASTER)
              .appName("fleet-anomaly")
              .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
              .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
              .config("spark.sql.session.timeZone", "UTC")
              .config("spark.sql.execution.arrow.pyspark.enabled", "true")
              .config("spark.ui.showConsoleProgress", "false")
              .config("spark.sql.warehouse.dir", str(config.OUT_DIR / "spark-warehouse"))
              .getOrCreate())
    _spark.sparkContext.setLogLevel("ERROR")
    return _spark


def stop() -> None:
    global _spark
    if _spark is not None:
        _spark.stop()
        _spark = None
