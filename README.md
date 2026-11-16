# experiments-ml

Reference code for machine-learning posts on [ingelenz.com](https://www.ingelenz.com): classical ML, data
engineering and MLOps at fleet scale. LLM experiments live in the sibling repository
[experiments-llm](https://github.com/vishalmkatkar/experiments-llm).

## Series

| Series | Code | Status |
|---|---|---|
| Fleet anomaly detection (6 parts): from unsupervised anomalies to failure diagnosis and service recommendations, with Apache Spark and MLflow | [`src/app/fleet_anomaly/`](src/app/fleet_anomaly/) | Synthetic fleet, ground truth and Spark/MLflow setup built. Part 1 drafted in [`blog/`](blog/); Parts 2–6 next |

## Layout

```
blog/        drafts of the posts, as standalone HTML
src/app/     one package per series; run from src/ with  python -m app.<package>.main
```

## Environment

Everything runs on one laptop: Spark in local mode, MLflow tracking to a local SQLite database.

- Python 3.11 in a `.venv` at the repository root
- Java 17 or 21 for Spark 4.x. The code uses a Homebrew JDK 21 (`brew install openjdk@21`) for its own process only,
  so your default Java can stay newer; point `FA_JAVA_HOME` elsewhere if your JDK lives somewhere else
- Dependencies pinned in [`requirements.txt`](requirements.txt)

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install openjdk@21

cd src
../.venv/bin/python -m app.fleet_anomaly.main --fast
```
