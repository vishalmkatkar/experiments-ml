"""Check a generated fleet with Spark, and describe it.

The checks make sure the tables honour their contracts (schema.TABLES) and that the
ground truth is consistent with what was observed. The description shows the fleet
the later parts will work on. Nothing here detects anything: this is the data, not a result.
"""

import json
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from .. import config
from ..schema import TABLES
from . import world as W

TRUTH = ["machines", "modes", "anomaly_windows", "failures"]


def load(spark: SparkSession, data_dir: Path) -> dict[str, DataFrame]:
    dfs = {name: spark.read.parquet(str(data_dir / f"{name}.parquet")) for name in TABLES}
    dfs.update({f"truth_{n}": spark.read.parquet(str(data_dir / "truth" / f"{n}.parquet")) for n in TRUTH})
    return dfs


def check(results: list, name: str, ok: bool, detail: str) -> None:
    results.append({"check": name, "ok": bool(ok), "detail": detail})


def validate(spark: SparkSession, data_dir: Path) -> dict:
    t0 = time.perf_counter()
    summary = json.loads((data_dir / "truth" / "summary.json").read_text())
    hours = summary["days"] * 24
    d = load(spark, data_dir)
    results: list[dict] = []

    # Contracts: column names and types (pandas writes everything nullable, so nullability is checked as data).
    for name, contract in TABLES.items():
        got = {f.name: f.dataType for f in d[name].schema.fields}
        want = {f.name: f.dataType for f in contract.fields}
        check(results, f"schema:{name}", got == want, "matches contract" if got == want else f"got {got}")
        required = [f.name for f in contract.fields if not f.nullable]
        nulls = d[name].select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in required]).first()
        bad = {c: nulls[c] for c in required if nulls[c]}
        check(results, f"nulls:{name}", not bad, "no nulls in required columns" if not bad else str(bad))

    tel = d["telemetry"]
    rows = tel.count()
    n_machines = d["machines"].count()
    check(results, "telemetry:rows", rows == summary["telemetry_rows"], f"{rows:,} rows")
    nan = tel.select([F.sum(F.isnan(s).cast("int")).alias(s) for s in config.SENSORS]).first().asDict()
    check(results, "telemetry:no_nan", not any(nan.values()), "no NaN sensor values")
    dupes = rows - tel.select("machine_id", "ts").distinct().count()
    check(results, "telemetry:unique_hours", dupes == 0, f"{dupes} duplicate (machine, hour) rows")
    per_machine = tel.groupBy("machine_id").count()
    missing = per_machine.select(F.sum(F.lit(hours) - F.col("count"))).first()[0]
    check(results, "telemetry:missing_hours", missing == summary["missing_hours"],
          f"{missing:,} missing machine-hours ({missing / (n_machines * hours):.2%})")
    gap = (tel.withColumn("prev", F.lag("ts").over(Window.partitionBy("machine_id").orderBy("ts")))
           .withColumn("gap_h", (F.col("ts").cast("long") - F.col("prev").cast("long")) / 3600)
           .filter(F.col("gap_h") > 1))
    longest = gap.agg(F.max("gap_h")).first()[0] or 0
    check(results, "telemetry:gaps", True, f"{gap.count():,} gaps, longest {longest - 1:.0f} h missing")

    codes = {r.code for r in d["error_logs"].select("code").distinct().collect()}
    check(results, "logs:known_codes", codes <= set(W.ERROR_CODES), f"{len(codes)} codes, all known")

    # Ground truth against observations.
    tf = d["truth_failures"]
    n_fail = tf.count()
    check(results, "failures:match_truth", n_fail == d["failures"].count() == summary["failures"], f"{n_fail} failures")
    late = tf.filter((F.col("logged_ts") < F.col("true_ts")) |
                     (F.col("logged_ts") > F.col("true_ts") + F.expr(f"INTERVAL {W.LOG_DELAY_HOURS[1]} HOURS")))
    check(results, "failures:logged_late", late.count() == 0,
          f"logged 0-{W.LOG_DELAY_HOURS[1]} h after the true failure")
    pre = d["truth_anomaly_windows"].filter(F.col("kind") == "failure_precursor")
    no_pre = tf.join(pre, "failure_id", "left_anti").count()
    check(results, "failures:precursor", no_pre == 0, "every failure has a precursor window")
    rep = (tf.join(d["service_records"].select("service_id", F.col("ts").alias("svc_ts")), "service_id")
           .filter(F.col("svc_ts") < F.col("true_ts")))
    check(results, "service:after_failure", rep.count() == 0 and tf.filter(F.col("service_id").isNull()).count() == 0,
          "every failure has a repair visit after it")
    rp = tf.filter("is_repeat")
    bad_rp = (rp.alias("r").join(tf.alias("o"), F.col("r.repeat_of") == F.col("o.failure_id"))
              .filter(F.col("o.fix_held")).count())
    check(results, "failures:repeats", bad_rp == 0, f"{rp.count()} repeat failures, each after a fix that didn't hold")

    # Profiles never appear in observed tables.
    leaked = [n for n in TABLES if "profile" in d[n].columns]
    check(results, "truth:hidden", not leaked, "no observed table names the profile")

    # Description: how many error logs fall inside a failure precursor? The rest is noise.
    w = d["truth_anomaly_windows"]
    logs = d["error_logs"].withColumn("log_id", F.monotonically_increasing_id())
    inside = (logs.alias("l").join(w.alias("w"), (F.col("l.machine_id") == F.col("w.machine_id")) &
                                   F.col("l.ts").between(F.col("w.start_ts"), F.col("w.end_ts")))
              .groupBy("l.log_id").agg(F.max(F.when(F.col("w.kind") == "failure_precursor", 1).otherwise(0))
                                       .alias("in_failure")))
    n_logs = logs.count()
    in_failure = inside.filter("in_failure = 1").count()

    # Description: typical sensor levels per hidden profile and mode (normal_load), from Spark.
    joined = tel.join(d["truth_modes"].select("machine_id", "ts", "mode"), ["machine_id", "ts"])
    joined = joined.join(d["truth_machines"].select("machine_id", "profile"), "machine_id")
    levels = (joined.groupBy("profile", "mode")
              .agg(F.count("*").alias("hours"), *(F.round(F.avg(s), 2).alias(s) for s in config.SENSORS))
              .orderBy("profile", "mode").toPandas())

    return {
        "data_dir": str(data_dir), "checks": results, "passed": all(r["ok"] for r in results),
        "fleet": {"machines": n_machines, "telemetry_rows": rows, "error_logs": n_logs,
                  "error_logs_in_failure_precursors": in_failure,
                  "error_log_noise_share": round(1 - in_failure / max(n_logs, 1), 4), "failures": n_fail},
        "levels": levels.to_dict(orient="records"),
        "validate_seconds": round(time.perf_counter() - t0, 1),
    }


def print_report(report: dict, quiet: bool = False) -> None:
    import pandas as pd

    for r in report["checks"]:
        if not quiet or not r["ok"]:
            print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['check']:<28} {r['detail']}")
    f = report["fleet"]
    print(f"\n  {f['machines']} machines, {f['telemetry_rows']:,} telemetry rows, {f['failures']} failures, "
          f"{f['error_logs']:,} error logs ({f['error_log_noise_share']:.0%} outside any failure precursor)")
    if not quiet:
        lv = pd.DataFrame(report["levels"])
        lv = lv[lv["mode"].isin(["idle", "normal_load", "peak"])]
        print("\n  Mean sensor levels by hidden profile and mode (ground truth; the pipeline never sees these labels):")
        print("  " + lv.to_string(index=False).replace("\n", "\n  "))
