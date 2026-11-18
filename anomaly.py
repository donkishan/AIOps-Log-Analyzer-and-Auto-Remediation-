"""Step 5 - Time-series anomaly detection + crash prediction on server metrics.

Idea: Prophet learns what "normal" looks like (trend + daily seasonality) from history,
then forecasts an expected band. Points outside the band = anomalies.
We also extrapolate the recent trend to predict WHEN a resource will hit its limit
(e.g. "payment-api memory will hit 95% in ~25 min") -> fix it BEFORE the crash.

Run:  python anomaly.py
"""
import logging

import numpy as np
import pandas as pd

from config import METRICS, SERVICES
from ingest import get_metrics

try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:  # Prophet failed to install -> fall back gracefully
    HAS_PROPHET = False
logging.getLogger("cmdstanpy").disabled = True   # silence Stan's chatty logs
logging.getLogger("prophet").setLevel(logging.WARNING)

LIMITS = {"cpu_pct": 95, "mem_pct": 95, "disk_pct": 95, "latency_ms": 1000}


def _load(service, metric):
    df = get_metrics(service, metric)[["timestamp", "value"]]
    return df.rename(columns={"timestamp": "ds", "value": "y"})


def detect(service: str, metric: str, method: str = "prophet",
           holdout_hours: int = 6, forecast_minutes: int = 120) -> pd.DataFrame:
    """Return ds, y, yhat, yhat_lower, yhat_upper, anomaly (bool)."""
    df = _load(service, metric)
    if df.empty:
        return df
    cutoff = df["ds"].max() - pd.Timedelta(hours=holdout_hours)
    train = df[df["ds"] < cutoff]   # learn "normal" only from the past

    if method == "prophet" and HAS_PROPHET:
        m = Prophet(daily_seasonality=True, weekly_seasonality=False,
                    yearly_seasonality=False, interval_width=0.99,
                    changepoint_prior_scale=0.01)
        m.fit(train)
        future = m.make_future_dataframe(periods=forecast_minutes // 5, freq="5min")
        # make_future_dataframe continues from the train end; make sure we cover all real points too
        future = pd.concat([df[["ds"]], future]).drop_duplicates().sort_values("ds")
        fc = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    else:
        # Seasonal baseline: mean +/- 4 std for each 5-min slot of the day
        slot = lambda s: s.dt.hour * 12 + s.dt.minute // 5
        prof = train.assign(slot=slot(train["ds"])).groupby("slot")["y"].agg(["mean", "std"])
        fc = df[["ds"]].copy()
        fc["slot"] = slot(fc["ds"])
        fc = fc.join(prof, on="slot")
        fc["yhat"] = fc["mean"]
        fc["yhat_lower"] = fc["mean"] - 4 * fc["std"]
        fc["yhat_upper"] = fc["mean"] + 4 * fc["std"]
        fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    out = fc.merge(df, on="ds", how="left")
    out["anomaly"] = (out["y"] > out["yhat_upper"]) | (out["y"] < out["yhat_lower"])
    out["anomaly"] = out["anomaly"] & out["y"].notna()
    return out


def predict_breach(service: str, metric: str, lookback_minutes: int = 60):
    """Fit a line on the recent trend -> minutes until the metric crosses its limit."""
    df = _load(service, metric)
    recent = df[df["ds"] >= df["ds"].max() - pd.Timedelta(minutes=lookback_minutes)]
    if len(recent) < 3:
        return None
    t = (recent["ds"] - recent["ds"].min()).dt.total_seconds() / 60
    slope, intercept = np.polyfit(t, recent["y"], 1)
    current, limit = float(recent["y"].iloc[-1]), LIMITS[metric]
    if current >= limit:
        return {"current": current, "limit": limit, "slope_per_min": slope, "eta_min": 0}
    if slope <= 0:
        return {"current": current, "limit": limit, "slope_per_min": slope, "eta_min": None}
    return {"current": current, "limit": limit, "slope_per_min": round(slope, 3),
            "eta_min": round((limit - current) / slope, 1)}


def scan_all(method: str = "prophet", recent_minutes: int = 60) -> pd.DataFrame:
    """Check every service x metric, report the ones misbehaving right now."""
    rows = []
    for svc in SERVICES:
        for metric in METRICS:
            res = detect(svc, metric, method=method)
            if res.empty:
                continue
            last = res[res["y"].notna()]
            recent = last[last["ds"] >= last["ds"].max() - pd.Timedelta(minutes=recent_minutes)]
            n_anom = int(recent["anomaly"].sum())
            if n_anom >= 3:
                breach = predict_breach(svc, metric) or {}
                rows.append({
                    "service": svc, "metric": metric, "anomalous_points": n_anom,
                    "current": round(float(recent["y"].iloc[-1]), 1),
                    "expected": round(float(recent["yhat"].iloc[-1]), 1),
                    "limit": LIMITS[metric], "eta_to_limit_min": breach.get("eta_min"),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    method = "prophet" if HAS_PROPHET else "seasonal"
    print(f"Scanning all service/metric pairs with method = {method} ...\n")
    print(scan_all(method).to_string(index=False))
    print("\nCrash prediction for payment-api memory:", predict_breach("payment-api", "mem_pct"))
