"""Step 4 - Cluster error logs and flag anomalous clusters.

Pipeline:
  raw message --mask--> template --TF-IDF--> vector --DBSCAN--> cluster
  then: compare each cluster's recent rate vs its historical baseline.

Why masking? "pod payment-api-3fa9c2d1" and "pod payment-api-77be01aa" are the SAME
problem. Replacing IDs/IPs/numbers with tokens turns 10,000 lines into ~20 templates.

Run:  python clustering.py
"""
import re

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from ingest import get_logs

MASKS = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<IP>"),
    (re.compile(r"(?<=[_\-])[0-9a-f]{6,}\b"), "<ID>"),
    (re.compile(r"\b\d+(?:\.\d+)?"), "<NUM>"),
]


def mask_message(msg: str) -> str:
    for pattern, token in MASKS:
        msg = pattern.sub(token, msg)
    return msg


def cluster_logs(df: pd.DataFrame, eps: float = 0.5, min_count: int = 5,
                 recent_minutes: int = 60):
    """Returns (logs_with_cluster, cluster_summary, template_points_for_plot)."""
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame()
    df = df.copy()
    df["template"] = df["message"].map(mask_message)

    # cluster UNIQUE templates (fast), weighted by how often each appears
    tpl = df.groupby("template").size().rename("count").reset_index()
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z<>_/.:]+", ngram_range=(1, 2))
    X = vec.fit_transform(tpl["template"])
    labels = DBSCAN(eps=eps, min_samples=min_count, metric="cosine").fit(
        X, sample_weight=tpl["count"]).labels_
    tpl["cluster"] = labels

    # 2-D coordinates so we can draw the clusters in the UI
    if X.shape[1] > 2 and len(tpl) > 2:
        xy = TruncatedSVD(n_components=2, random_state=0).fit_transform(X)
        tpl["x"], tpl["y"] = xy[:, 0], xy[:, 1]
    else:
        tpl["x"], tpl["y"] = np.arange(len(tpl)), 0

    df = df.merge(tpl[["template", "cluster"]], on="template")

    # ---------------- anomaly scoring per cluster
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    t_max = df["timestamp"].max()
    t_min = df["timestamp"].min()
    cutoff = t_max - pd.Timedelta(minutes=recent_minutes)
    base_minutes = max((cutoff - t_min).total_seconds() / 60, 1)

    rows = []
    for cid, g in df.groupby("cluster"):
        recent = (g["timestamp"] >= cutoff).sum()
        base = (g["timestamp"] < cutoff).sum()
        rate_recent = recent / recent_minutes
        rate_base = base / base_minutes
        spike = (rate_recent + 1e-3) / (rate_base + 1e-3)
        is_new = base == 0 and recent >= min_count
        anomalous = bool(cid != -1 and (is_new or (spike >= 5 and recent >= min_count)))
        top_tpl = g["template"].value_counts().index[0]
        rows.append({
            "cluster": int(cid),
            "label": "rare / noise" if cid == -1 else top_tpl[:90],
            "template": top_tpl,
            "count": len(g),
            "recent_count": int(recent),
            "spike_ratio": round(float(spike), 1),
            "is_new": bool(is_new),
            "anomalous": anomalous,
            "level": g["level"].mode()[0],
            "services": ", ".join(sorted(g["service"].unique())),
            "hosts": ", ".join(sorted(g["host"].unique())),
            "first_seen": g["timestamp"].min(),
            "last_seen": g["timestamp"].max(),
            "samples": g.sort_values("timestamp")["message"].tail(5).tolist(),
        })
    summary = (pd.DataFrame(rows)
               .sort_values(["anomalous", "spike_ratio"], ascending=False)
               .reset_index(drop=True))
    tpl = tpl.merge(summary[["cluster", "anomalous"]], on="cluster", how="left")
    return df, summary, tpl


SEVERITY = {"WARN": 1, "ERROR": 2, "CRITICAL": 3}


def group_incidents(summary: pd.DataFrame) -> list[dict]:
    """Event correlation: anomalous clusters from the same service = ONE incident.
    (OOMKilled + OutOfMemoryError + Back-off restarting are 3 symptoms of 1 problem.)"""
    if summary.empty:
        return []
    incidents = []
    for svc, g in summary[summary["anomalous"]].groupby("services"):
        incidents.append({
            "services": svc,
            "hosts": ", ".join(sorted({h for hs in g["hosts"] for h in hs.split(", ")})),
            "clusters": g["cluster"].tolist(),
            "templates": g["template"].tolist(),
            "level": max(g["level"], key=lambda l: SEVERITY.get(l, 0)),
            "recent_count": int(g["recent_count"].sum()),
            "spike_ratio": float(g["spike_ratio"].max()),
            "first_seen": g["first_seen"].min(),
            "samples": [s for ss in g["samples"] for s in ss[-3:]],
        })
    return sorted(incidents, key=lambda i: -SEVERITY.get(i["level"], 0) * 1e6 - i["recent_count"])


def detect_log_anomalies(**kw):
    errors = get_logs(levels=["WARN", "ERROR", "CRITICAL"])
    return cluster_logs(errors, **kw)


if __name__ == "__main__":
    print("Masking example:")
    m = "OOMKilled: container payment-api in pod payment-api-3fa9c2d1 exceeded memory limit 512Mi"
    print("  raw     :", m, "\n  template:", mask_message(m), "\n")
    _, summary, _ = detect_log_anomalies()
    pd.set_option("display.width", 200, "display.max_colwidth", 80)
    print(summary[["cluster", "anomalous", "count", "recent_count", "spike_ratio", "services", "label"]]
          .to_string(index=False))
    print("\nCorrelated incidents:")
    for i, inc in enumerate(group_incidents(summary), 1):
        print(f"  {i}. [{inc['level']}] {inc['services']}: {len(inc['templates'])} clusters, "
              f"{inc['recent_count']} errors in last hour")
