# experiments-ml

Reference code for machine-learning posts on [ingelenz.com](https://www.ingelenz.com): classical ML, data
engineering and MLOps at fleet scale. LLM experiments live in the sibling repository
[experiments-llm](https://github.com/vishalmkatkar/experiments-llm).

## Series

| Series | Code | Status |
|---|---|---|
| Fleet anomaly detection (6 parts): from unsupervised anomalies to failure diagnosis and service recommendations, with Apache Spark and MLflow | [`src/app/fleet_anomaly/`](src/app/fleet_anomaly/) | Planned. Part 1 drafted in [`blog/`](blog/) |

## Layout

```
blog/        drafts of the posts, as standalone HTML
src/app/     one package per series; run from src/ with  python -m app.<package>.main
```

## Environment

Everything runs on one laptop: Spark in local mode, MLflow tracking to a local SQLite database.

- Python 3.11 in a `.venv` at the repository root
- Java 17 or 21 for Spark (Spark does not support Java 19)
- Dependencies in [`requirements.txt`](requirements.txt), pinned once the environment is set up
