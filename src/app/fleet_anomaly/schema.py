"""Contracts between the stages: the observed tables, and the episode record.

The table schemas are what the pipeline may rely on. Ground-truth tables have no
contract here on purpose: no pipeline stage reads them, only evaluation does.

The episode is the one object the stages pass along (Part 1, "The episode record").
Detect creates it; Correlate, Diagnose and Recommend each fill in their own fields.
Everything after `deviations` is optional, so an episode with no known cause stops at
"unusual, no known cause" instead of guessing.
"""

import json
from dataclasses import asdict, dataclass, field

from pyspark.sql.types import (BooleanType, DateType, DoubleType, FloatType, StringType, StructField, StructType,
                               TimestampType)

from . import config


def _struct(*fields: tuple[str, object, bool]) -> StructType:
    return StructType([StructField(n, t, nullable) for n, t, nullable in fields])


TABLES: dict[str, StructType] = {
    "machines": _struct(("machine_id", StringType(), False), ("model", StringType(), False),
                        ("site", StringType(), False), ("install_date", DateType(), False),
                        ("rated_kw", DoubleType(), False)),
    "telemetry": _struct(("machine_id", StringType(), False), ("ts", TimestampType(), False),
                         *((s, FloatType(), False) for s in config.SENSORS)),
    "error_logs": _struct(("machine_id", StringType(), False), ("ts", TimestampType(), False),
                          ("code", StringType(), False), ("severity", StringType(), False)),
    "failures": _struct(("failure_id", StringType(), False), ("machine_id", StringType(), False),
                        ("ts", TimestampType(), False), ("component", StringType(), False),
                        ("failure_mode", StringType(), False)),
    "service_records": _struct(("service_id", StringType(), False), ("machine_id", StringType(), False),
                               ("ts", TimestampType(), False), ("visit_type", StringType(), False),
                               ("note", StringType(), False), ("action", StringType(), False),
                               ("parts", StringType(), False), ("fix_held", BooleanType(), True)),
}


@dataclass
class Action:
    action: str
    first_time_fix: float
    cases: list[str] = field(default_factory=list)  # service_ids behind the ranking


@dataclass
class Episode:
    # Detect (Part 3)
    episode_id: str
    machine_id: str
    profile: str
    mode: str
    start: str
    end: str
    score_pct: float
    deviations: dict[str, str] = field(default_factory=dict)  # sensor -> "+4.1 MAD, rising"
    # Correlate (Part 4)
    broken_pairs: list[list[str]] | None = None
    codes: list[str] | None = None
    # Diagnose (Part 5)
    likely_mode: str | None = None
    confidence: float | None = None
    similar_episodes: list[str] | None = None
    typical_lead_days: float | None = None
    # Recommend (Part 6)
    actions: list[Action] | None = None

    @property
    def stage(self) -> str:
        """How far along the pipeline this episode got."""
        if self.actions is not None:
            return "recommended"
        if self.likely_mode is not None:
            return "diagnosed"
        if self.broken_pairs is not None or self.codes is not None:
            return "correlated"
        return "detected"

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})

    @classmethod
    def from_json(cls, text: str) -> "Episode":
        d = json.loads(text)
        if d.get("actions") is not None:
            d["actions"] = [Action(**a) for a in d["actions"]]
        return cls(**d)
