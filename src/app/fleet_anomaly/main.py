"""Build and check the synthetic fleet the series runs on.

    cd src
    ../.venv/bin/python -m app.fleet_anomaly.main                  # 200 machines x 1 year: generate if missing, validate
    ../.venv/bin/python -m app.fleet_anomaly.main --fast           # 30 machines x 120 days, for quick checks
    ../.venv/bin/python -m app.fleet_anomaly.main --generate       # rebuild even if the data exists
    ../.venv/bin/python -m app.fleet_anomaly.main --machines 50 --days 90 --seed 7
    ../.venv/bin/python -m app.fleet_anomaly.main --no-mlflow      # skip the MLflow run

Later parts add their stages here (profiles, detect, correlate, diagnose, recommend).
"""

import argparse
import json
import sys
import time

from . import config


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", action="store_true", help=f"{config.FAST_MACHINES} machines x {config.FAST_DAYS} days")
    p.add_argument("--machines", type=int)
    p.add_argument("--days", type=int)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--generate", action="store_true", help="rebuild the data even if it exists")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    n = args.machines or (config.FAST_MACHINES if args.fast else config.N_MACHINES)
    days = args.days or (config.FAST_DAYS if args.fast else config.N_DAYS)

    from .data.generate import dataset_dir, generate

    data_dir = dataset_dir(n, days, args.seed)
    summary_path = data_dir / "truth" / "summary.json"
    if args.generate or not summary_path.exists():
        print(f"Generating {n} machines x {days} days (seed {args.seed}) ...", flush=True)
        summary = generate(n, days, args.seed, data_dir)
        print(f"  {summary['telemetry_rows']:,} telemetry rows in {summary['generate_seconds']}s -> {data_dir}")
    else:
        summary = json.loads(summary_path.read_text())
        print(f"Using {data_dir}")

    from .data.validate import print_report, validate
    from .spark import get_spark, stop

    t0 = time.perf_counter()
    spark = get_spark()
    print(f"\nValidating with Spark {spark.version} ({config.SPARK_MASTER}, started in "
          f"{time.perf_counter() - t0:.1f}s) ...", flush=True)
    report = validate(spark, data_dir)
    print_report(report, args.quiet)
    print(f"\n  Validation {'passed' if report['passed'] else 'FAILED'} in {report['validate_seconds']}s")

    report_path = config.OUT_DIR / f"{data_dir.name}.validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))

    if not args.no_mlflow:
        from . import tracking

        with tracking.start_run("generate", tags={"part": "1", "stage": "data"}) as run:
            import mlflow

            mlflow.log_params({"seed": args.seed, "machines": n, "days": days, "start": config.START,
                               "spark": spark.version})
            tracking.log_dict_flat("", {k: summary[k] for k in ("telemetry_rows", "missing_hours", "error_logs",
                                                                 "failures", "repeat_failures", "failures_by_mode",
                                                                 "generate_seconds")})
            tracking.log_dict_flat("", {"validate_seconds": report["validate_seconds"],
                                        "error_log_noise_share": report["fleet"]["error_log_noise_share"],
                                        "checks_failed": sum(not r["ok"] for r in report["checks"])})
            tracking.log_file(summary_path)
            tracking.log_file(report_path)
            print(f"  MLflow run {run.info.run_id} in experiment '{config.MLFLOW_EXPERIMENT}'")
    stop()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
