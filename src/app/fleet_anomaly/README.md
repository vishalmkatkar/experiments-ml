# Fleet anomaly detection

Reference code for the six-part series that starts with
[Anomalies Are Not Failures](../../../blog/fleet-anomaly-part-1-anomalies-are-not-failures.html).

Not built yet. Planned modules, in build order:

| Module | Part | Contents |
|---|---|---|
| `data` | all | Synthetic fleet generator (about 200 machines, 4 hidden profiles, 6 sensors, one year hourly), logs, failures, service records, ground truth |
| `profiles` | 2 | Machine features, profile and operating-mode clustering |
| `detect` | 3 | Per-profile detectors, per-machine calibration, episodes, scoring wrapper |
| `correlate` | 4 | Correlation structure, break score, log linkage, association rules |
| `diagnose` | 5 | Episode signatures, failure-mode matching, evaluation |
| `recommend` | 6 | Case base, ALS, outcome-weighted ranking, LLM work orders |

Part 1 is published only once `data`, `profiles` and `detect` run end to end.
