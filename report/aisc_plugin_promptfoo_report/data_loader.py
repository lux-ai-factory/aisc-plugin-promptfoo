from __future__ import annotations

from typing import Any

from sqlalchemy import Float, cast, func, select
from sqlalchemy.orm import Session

from resources.sql_alchemy import Measurement, Metric, Observation


# Per-test metrics emitted by the Promptfoo plugin under the
# arbitrary-measure-dimensions architecture. Each Measurement row has
# dimensions = {mode, category, strategy}.
PER_TEST_METRICS = ("success", "score", "latency_ms", "cost", "refusal")

UNIT_BY_AGGREGATE = {
    "pass_rate": "ratio",
    "fail_rate": "ratio",
    "refusal_rate": "ratio",
    "mean_score": "score",
    "mean_latency_ms": "ms",
    "total_cost": "USD",
    "n_tests": "tests",
}


class PromptfooDataLoader:
    def __init__(self, db_session: Session | None = None):
        if not db_session:
            raise RuntimeError("No database session available")
        self.session = db_session

    def _scores_by_metric(self) -> dict[str, list[float]]:
        stmt = (
            select(
                Metric.name.label("metric_name"),
                cast(Measurement.score, Float).label("score"),
            )
            .join(Measurement, Measurement.metric_id == Metric.id)
            .join(Observation, Measurement.observation_id == Observation.id)
            .where(Metric.name.in_(PER_TEST_METRICS))
        )
        rows = self.session.execute(stmt).mappings().all()
        out: dict[str, list[float]] = {name: [] for name in PER_TEST_METRICS}
        for row in rows:
            v = row["score"]
            if v is None:
                continue
            out.setdefault(row["metric_name"], []).append(float(v))
        return out

    def _rows_with_dims(self, metric_name: str) -> list[tuple[float, dict]]:
        stmt = (
            select(
                cast(Measurement.score, Float).label("score"),
                Measurement.dimensions.label("dimensions"),
            )
            .join(Metric, Measurement.metric_id == Metric.id)
            .where(Metric.name == metric_name)
        )
        rows = self.session.execute(stmt).mappings().all()
        return [(float(r["score"]), r["dimensions"] or {}) for r in rows if r["score"] is not None]

    def _avg(self, vals: list[float]) -> float | None:
        return (sum(vals) / len(vals)) if vals else None

    def _sum(self, vals: list[float]) -> float | None:
        return sum(vals) if vals else None

    def _group_avg(self, rows: list[tuple[float, dict]], key: str) -> list[dict[str, Any]]:
        buckets: dict[Any, list[float]] = {}
        for score, dims in rows:
            buckets.setdefault(dims.get(key, "unknown"), []).append(score)
        return [
            {key: k, "n": len(v), "rate": sum(v) / len(v)}
            for k, v in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        ]

    def _n_observations(self) -> int:
        stmt = (
            select(func.count(func.distinct(Observation.id)))
            .join(Measurement, Measurement.observation_id == Observation.id)
            .join(Metric, Measurement.metric_id == Metric.id)
            .where(Metric.name.in_(PER_TEST_METRICS))
        )
        return int(self.session.scalar(stmt) or 0)

    def compute_statistics(self) -> dict[str, Any]:
        scores = self._scores_by_metric()
        n_obs = self._n_observations()

        success_rows = self._rows_with_dims("success")
        score_rows = self._rows_with_dims("score")
        latency_rows = self._rows_with_dims("latency_ms")
        cost_rows = self._rows_with_dims("cost")
        refusal_rows = self._rows_with_dims("refusal")

        pass_rate = self._avg(scores.get("success", []))
        mean_score = self._avg(scores.get("score", []))
        mean_latency_ms = self._avg(scores.get("latency_ms", []))
        total_cost = self._sum(scores.get("cost", []))
        refusal_rate = self._avg(scores.get("refusal", []))
        n_tests = len(success_rows) or len(score_rows) or len(latency_rows)

        breakdowns = {
            "pass_by_category": self._group_avg(success_rows, "category"),
            "pass_by_strategy": self._group_avg(success_rows, "strategy"),
            "pass_by_mode": self._group_avg(success_rows, "mode"),
            "refusal_by_category": self._group_avg(refusal_rows, "category"),
        }

        summary = {
            "evaluations": n_obs,
            "n_tests": int(n_tests) if n_tests else None,
            "pass_rate": pass_rate,
            "fail_rate": (1.0 - pass_rate) if pass_rate is not None else None,
            "refusal_rate": refusal_rate,
            "mean_score": mean_score,
            "mean_latency_ms": mean_latency_ms,
            "total_cost": total_cost,
        }

        metrics_table = [
            {
                "name": name,
                "value": summary.get(name),
                "unit": UNIT_BY_AGGREGATE.get(name, ""),
                "n_observations": n_obs,
            }
            for name in (
                "pass_rate",
                "fail_rate",
                "refusal_rate",
                "mean_score",
                "mean_latency_ms",
                "total_cost",
                "n_tests",
            )
        ]

        return {
            "nObservations": n_obs,
            "summary": summary,
            "breakdowns": breakdowns,
            "metricsTable": metrics_table,
        }
