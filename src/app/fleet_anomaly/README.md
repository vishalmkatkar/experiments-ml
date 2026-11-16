# Fleet anomaly detection

Reference code for the six-part series that starts with
[Anomalies Are Not Failures](../../../blog/fleet-anomaly-part-1-anomalies-are-not-failures.html).

Part 1 is the architecture. This package starts with what every later part depends on: a synthetic fleet with
ground truth, the contracts between the stages (table schemas and the episode record), and a local Spark and
MLflow setup. The pipeline stages arrive one part at a time.

## Run

```bash
pip install -r requirements.txt     # from the repo root, into .venv
brew install openjdk@21             # Spark 4.x needs Java 17 or 21; used by this process only

cd src
../.venv/bin/python -m app.fleet_anomaly.main --fast        # 30 machines x 120 days, ~30 s
../.venv/bin/python -m app.fleet_anomaly.main               # 200 machines x 1 year: generate if missing, validate
../.venv/bin/python -m app.fleet_anomaly.main --generate    # rebuild even if the data exists
../.venv/bin/python -m app.fleet_anomaly.main --machines 50 --days 90 --seed 7
../.venv/bin/python -m app.fleet_anomaly.main --no-mlflow

../.venv/bin/mlflow ui --backend-store-uri sqlite:///../mlflow.db   # browse the runs
```

Each run generates the fleet (unless it already exists), checks it with Spark, prints a description, writes a
validation report to `out/`, and logs the run to the MLflow experiment `fleet-anomaly`. Generated data goes to
`data/generated/fleet_<machines>m_<days>d_seed<seed>/` and is not committed; the same seed always produces the same
tables (`truth/summary.json` holds a hash of each one). [`samples/fleet_summary.json`](samples/fleet_summary.json) is
the summary of the default fleet, for comparison.

Settings live in [`config.py`](config.py), each with an `FA_*` environment override (`FA_JAVA_HOME`, `FA_SEED`,
`FA_MACHINES`, `FA_SPARK_DRIVER_MEMORY`, ...).

## The synthetic fleet

Declared as data in [`data/world.py`](data/world.py), generated with numpy in
[`data/generate.py`](data/generate.py): 200 pumps, compressors and fans at five sites, hourly readings from six
sensors (vibration, temperature, pressure, motor current, flow, speed) for 2026.

**Observed tables**, what a real pipeline would get:

| Table | Grain | Columns |
|---|---|---|
| `machines` | machine | `machine_id`, `model`, `site`, `install_date`, `rated_kw` |
| `telemetry` | machine × hour | `machine_id`, `ts`, six sensors; some hours missing |
| `error_logs` | event | `machine_id`, `ts` (controller clock), `code`, `severity` |
| `failures` | event | `failure_id`, `machine_id`, `ts` (when logged), `component`, `failure_mode` |
| `service_records` | visit | `service_id`, `machine_id`, `ts`, `visit_type`, free-text `note`, `action`, `parts`, `fix_held` |

**Ground truth**, in `truth/`, for measuring only: each machine's hidden profile, the operating mode and load of
every hour, every anomaly window with its cause, and every failure with its true onset, true time, logged time,
repair, the action taken, the habitual action and the best action.

How it behaves:

- **Four hidden profiles.** Steady pumps around the clock, heavy compressors on two shifts, high-speed fans on day
  shift, batch machines with irregular runs. Profiles differ in sensor levels, how strongly each sensor responds to
  load, and the lag between motor current and temperature. Two nominal models, each spread across two profiles.
- **Operating modes from schedules.** Idle, startup, normal load and peak, driven by each site's shifts. Startups
  spike vibration and current; peaks run hot. Temperature also follows site climate, season and time of day.
- **Four failure modes, each with a signature.**

  | Failure mode | Warning | Signature | Codes |
  |---|---|---|---|
  | `bearing_wear` | 6–16 days | vibration rises, then temperature | E214, E031 |
  | `seal_leak` | 8–21 days | pressure falls while flow barely moves: the pressure–flow relationship breaks | E120 |
  | `winding_overheat` | 4–12 days | temperature climbs without more current: the current–temperature relationship breaks | E305, E031 |
  | `valve_failure` | 2–12 hours | pressure oscillates, flow drops | E410, E120 |

- **Repairs and outcomes.** Each failure leads to downtime and a repair visit. The action is picked the way
  technicians pick it, by habit; whether it holds is drawn from the action's true first-time fix rate, and a fix that
  doesn't hold brings the same failure back within 5–25 days.

### Traps planted on purpose

| Trap | What it looks like | Part that deals with it |
|---|---|---|
| Mode changes | startups, peaks and idle nights look unusual against a fleet-wide normal | 2, 3 |
| Model is not profile | same model number, different duty | 2 |
| Misfit machines | 3 machines fit no profile | 2 |
| Profile change | 3 machines are overhauled into batch duty mid-year | 2 |
| Benign anomalies | sensor glitches (spike, dropout, stuck), process upsets, site-wide heat waves; never followed by a failure, some raise the same codes failures do | 3, 4, 5 |
| Relationships break first | seal leaks and winding overheating show up in sensor pairs before single values | 4 |
| Late failure timestamps | logged 0–48 h after the true failure; degradation starts days earlier | 4, 5 |
| Controller clock skew | error-log times are off by up to ±90 min per machine | 4 |
| Noisy logs | most error codes have nothing to do with a failure | 4 |
| Post-repair reset | new parts run quieter for a week, which "predicts" the repair that already happened | 5 (leakage) |
| Habit is not what works | for 3 of 4 failure modes, the action picked most often isn't the one that holds | 6 |
| Free-text notes | same finding in different words, abbreviations, capitals and typos | 6 |

## How it maps to the blog

| Blog section (Part 1) | Code | Status |
|---|---|---|
| The data: five sources at different grains | [`data/`](data/), table contracts in [`schema.py`](schema.py) `TABLES` | Built |
| The episode record | [`schema.py`](schema.py) `Episode`: Detect fields required, later stages optional, JSON round-trip | Built |
| Spark and MLflow | [`spark.py`](spark.py) local session; [`tracking.py`](tracking.py) SQLite-backed MLflow | Built |
| The test bench | [`data/world.py`](data/world.py), [`data/generate.py`](data/generate.py), checked by [`data/validate.py`](data/validate.py) | Built |
| 1 Profile | `profiles` | Part 2 |
| 2 Detect | `detect` | Part 3 |
| 3 Correlate | `correlate` | Part 4 |
| 4 Diagnose | `diagnose` | Part 5 |
| 5 Recommend | `recommend` | Part 6 |

## The default fleet

Seed 42, 200 machines × 365 days, as generated:

| | |
|---|---|
| Machines by hidden profile | P1 60 · P2 59 · P3 39 · P4 39 · misfits 3 · profile changes 3 |
| Telemetry rows | 1,746,334 (5,666 machine-hours missing, 0.32%) |
| Hours by mode | idle 35.7% · startup 3.0% · normal load 42.6% · peak 18.4% · down 0.3% |
| Error logs | 57,696 |
| Failures | 129 (bearing 44 · seal 42 · winding 23 · valve 20), 33 of them repeats after a fix that didn't hold |
| Service visits | 129 repairs · 709 preventive · 89 no-fault inspections · 3 overhauls |
| Benign anomaly windows | 384 sensor glitches · 292 process upsets · 530 machine-level heat-wave windows |
| Generation time | about 3 s with numpy, ~60 MB of Parquet |
| Full run (`main.py --generate`) | 28 s on an M1 with 16 GB: Spark starts in ~5 s, validation takes 16 s |
| Error logs outside any failure precursor | 94% |
| Reproducible | a second generation with seed 42 gave identical hashes for all nine tables |

Spot check of the planted signatures: the median change over the last hours before each first failure, against
the two weeks before its onset.

| Failure mode | Vibration | Temperature | Pressure | Flow | Current | Pressure–flow correlation |
|---|---|---|---|---|---|---|
| bearing_wear | +142% | +17% | +2% | +1% | +3% | 0.99 → 0.99 |
| seal_leak | 0% | +3% | −24% | −4% | 0% | 0.99 → 0.97 |
| winding_overheat | +1% | +39% | +1% | +1% | +4% | 0.97 → 0.97 |
| valve_failure | +9% | 0% | +4% | −14% | +11% | 0.99 → 0.44 |

Observed first-time fix rates in the generated repairs (the habitual action first):

| Failure mode | Habitual action | Better action |
|---|---|---|
| bearing_wear | lubricate_and_recheck 9/21 (43%) | replace_bearing 13/14 (93%) |
| seal_leak | tighten_gland 9/20 (45%) | replace_seal 9/12 (75%) |
| winding_overheat | clean_cooling_fan 9/13 (69%) | rewind_motor 7/7 (100%) |
| valve_failure | replace_valve 18/18 (100%) | (the habit is right) |

`main.py` checks the tables with Spark: schemas against the contracts, required columns not null, no duplicate or NaN
readings, missing hours accounted for, failures logged late but never early, every failure with a precursor window
and a repair visit after it, every repeat failure after a fix that didn't hold, and no observed table naming a
profile. It also prints the mean sensor levels per hidden profile and mode, which shows why a single fleet-wide
"normal" can't work.

## Things learned building it

- **Spark 4 doesn't run on every Java.** It needs 17 or 21; this machine had 19 and 26. The session sets `JAVA_HOME`
  for its own process from `FA_JAVA_HOME` rather than changing the system default.
- **PySpark 4.3 warns about pandas 3.** pandas is pinned below 3, since the later parts lean on `applyInPandas`.
- **Spark can't read pandas' nanosecond timestamps.** Parquet is written with microsecond timestamps.
- **A small fleet can have no failures at all.** 20 machines × 60 days at 0.6 failures per machine-year produced
  none, and an empty table lost its columns. Every table now has explicit columns, and `--fast` is 30 × 120 days.
- **Truth links are easy to break silently.** A stale loop variable once gave every failure on a machine the same
  index, which unlinked most failures from their repair visits; it showed up only as every observed fix rate being
  100%. Validation now checks that every failure has a repair and every repeat follows a failed fix.
- **A broken relationship doesn't always show in a correlation.** Before a seal leak, pressure falls 24% while flow
  falls 4%, yet over a day the two still correlate at 0.97, because load drives both. Part 4's break score has to
  look at the relationship itself (the ratio, or the residual of one sensor predicted from the other), not only at
  the correlation coefficient.

## Limits

- **Synthetic data.** The signatures are cleaner than a real fleet's, and the fleet is generated from the same
  assumptions the pipeline will test. Each part says so and adds noise; a public predictive-maintenance dataset is
  planned as a second check.
- **Few failures per action.** 129 failures spread over 11 mode–action pairs leaves some pairs with 2 cases. Part 6
  has to report uncertainty, not just rates.
- **Spark runs locally.** Everything runs on one M1 with 16 GB in local mode. The code is what a cluster would run;
  the timings are a laptop's.
- **One repeat at most.** A repeat failure whose fix fails again is recorded with `fix_held` unknown rather than
  failing a third time.
