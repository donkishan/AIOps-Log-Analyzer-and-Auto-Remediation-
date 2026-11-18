"""Step 2 - Synthetic data generator.

Creates realistic server logs + metrics for 5 micro-services, with 4 real-world
incidents injected near the end (so we have something to detect and fix):

  1. payment-api   -> memory leak -> OOMKilled pods            (fix: restart / raise limit)
  2. nginx-gateway -> disk filling with logs -> "No space left" (fix: rotate/clean logs)
  3. redis-cache   -> cache timeouts / eviction storm          (fix: flush / restart redis)
  4. postgres-db   -> connection pool exhausted                (fix: Terraform max_connections)

Run:  python generate_logs.py
"""
import json
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import LOG_FILE, METRICS_FILE, SERVICES

random.seed(42)
np.random.seed(42)

HOSTS = {s: [f"{s}-node-{i}" for i in range(1, 3)] for s in SERVICES}

# ---------------------------------------------------------------- templates
NORMAL_INFO = [
    "GET /api/v1/{ep} 200 OK in {ms}ms",
    "POST /api/v1/{ep} 201 Created in {ms}ms",
    "Health check passed for {svc}",
    "User {uid} authenticated successfully",
    "Cache hit for key product:{num}",
    "Processed payment txn_{hex} amount={num}",
]
NORMAL_WARN = [
    "Slow query detected: SELECT * FROM orders took {ms}ms",
    "GET /api/v1/{ep} 404 Not Found",
    "Retrying request to {svc} (attempt {small}/3)",
]
NORMAL_ERROR = [  # rare background noise that is NOT an incident
    "Invalid JWT token from client {ip}",
    "Request timeout from upstream {ip} after {ms}ms",
]

INCIDENTS = {
    "oom": {
        "service": "payment-api", "level": "CRITICAL",
        "templates": [
            "OOMKilled: container payment-api in pod payment-api-{hex} exceeded memory limit 512Mi",
            "Back-off restarting failed container payment-api in pod payment-api-{hex}",
            "java.lang.OutOfMemoryError: Java heap space at PaymentProcessor.process(txn_{hex})",
        ],
    },
    "disk": {
        "service": "nginx-gateway", "level": "ERROR",
        "templates": [
            "No space left on device: write failed for /var/log/nginx/access.log (disk usage {pct}%)",
            "Failed to rotate log /var/log/nginx/error.log: No space left on device",
        ],
    },
    "redis": {
        "service": "redis-cache", "level": "ERROR",
        "templates": [
            "Redis timeout after 5000ms on GET session:{num}",
            "Redis OOM command not allowed when used memory > maxmemory, evicting keys",
        ],
    },
    "dbpool": {
        "service": "postgres-db", "level": "ERROR",
        "templates": [
            "Connection pool exhausted: unable to acquire connection to postgres-primary:5432 after 30000ms",
            "FATAL: sorry, too many clients already (max_connections=100) from {ip}",
        ],
    },
}
# how long before "now" each incident starts (minutes)
INCIDENT_WINDOWS = {"oom": 45, "disk": 35, "redis": 60, "dbpool": 40}


def _fill(t: str, svc: str) -> str:
    return t.format(
        ep=random.choice(["orders", "users", "cart", "checkout", "products"]),
        ms=random.randint(5, 900), svc=svc, uid=random.randint(1000, 99999),
        num=random.randint(1, 99999), hex=uuid.uuid4().hex[:8],
        small=random.randint(1, 3), ip=f"10.0.{random.randint(0,255)}.{random.randint(1,254)}",
        pct=random.randint(97, 100),
    )


def make_log(ts, svc, level, template):
    return {
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "service": svc,
        "host": random.choice(HOSTS[svc]),
        "level": level,
        "message": _fill(template, svc),
    }


def generate_logs(hours: int = 48, end: datetime | None = None) -> list[dict]:
    end = end or datetime.now().replace(microsecond=0)
    start = end - timedelta(hours=hours)
    logs, ts = [], start
    while ts < end:
        for svc in SERVICES:
            r = random.random()
            if r < 0.80:
                logs.append(make_log(ts, svc, "INFO", random.choice(NORMAL_INFO)))
            elif r < 0.95:
                logs.append(make_log(ts, svc, "WARN", random.choice(NORMAL_WARN)))
            elif r < 0.97:
                logs.append(make_log(ts, svc, "ERROR", random.choice(NORMAL_ERROR)))
        # incidents
        for key, win in INCIDENT_WINDOWS.items():
            if ts >= end - timedelta(minutes=win) and random.random() < 0.7:
                inc = INCIDENTS[key]
                for _ in range(random.randint(1, 3)):
                    logs.append(make_log(ts + timedelta(seconds=random.randint(0, 59)),
                                         inc["service"], inc["level"], random.choice(inc["templates"])))
        ts += timedelta(minutes=1)
    logs.sort(key=lambda x: x["timestamp"])
    return logs


def generate_metrics(days: int = 7, end: datetime | None = None) -> pd.DataFrame:
    """5-minute metrics with daily seasonality + incident-driven drift."""
    end = (end or datetime.now()).replace(second=0, microsecond=0)
    idx = pd.date_range(end=end, periods=days * 288, freq="5min")
    hour = idx.hour + idx.minute / 60
    daily = np.sin((hour - 6) / 24 * 2 * np.pi)  # peak mid-day traffic
    n = len(idx)
    rows = []
    for svc in SERVICES:
        cpu = 35 + 15 * daily + np.random.normal(0, 3, n)
        mem = 55 + 5 * daily + np.random.normal(0, 2, n)
        disk = 60 + np.linspace(0, 3, n) + np.random.normal(0, 0.3, n)
        lat = 120 + 40 * daily + np.random.normal(0, 10, n)

        def ramp(minutes):  # 0 -> 1 over the last `minutes`
            k = minutes // 5
            r = np.zeros(n)
            r[-k:] = np.linspace(0, 1, k)
            return r

        if svc == "payment-api":      # memory leak over last 3h
            r = ramp(180); mem += r * 36; lat += r * 400
        if svc == "nginx-gateway":    # disk filling over last 2h
            r = ramp(120); disk += r * 35
        if svc == "redis-cache":      # latency spike last 1h
            r = ramp(60); lat += r * 900; cpu += r * 30
        if svc == "postgres-db":      # cpu spike last 45min
            r = ramp(45); cpu += r * 50; lat += r * 600

        for m, vals in zip(["cpu_pct", "mem_pct", "disk_pct", "latency_ms"], [cpu, mem, disk, lat]):
            if m != "latency_ms":
                vals = np.clip(vals, 0, 100)
            rows.append(pd.DataFrame({"timestamp": idx, "service": svc, "metric": m,
                                      "value": np.round(vals, 2)}))
    return pd.concat(rows, ignore_index=True)


def generate_live_batch(seconds: int = 10, incident: str | None = None) -> list[dict]:
    """Small batch of 'right now' logs, used for live streaming / incident injection."""
    now = datetime.now().replace(microsecond=0)
    out = []
    for s in range(seconds):
        ts = now - timedelta(seconds=seconds - s)
        svc = random.choice(SERVICES)
        lvl, tpl = ("INFO", random.choice(NORMAL_INFO)) if random.random() < 0.85 else ("WARN", random.choice(NORMAL_WARN))
        out.append(make_log(ts, svc, lvl, tpl))
        if incident:
            inc = INCIDENTS[incident]
            out.append(make_log(ts, inc["service"], inc["level"], random.choice(inc["templates"])))
    return out


if __name__ == "__main__":
    end = datetime.now().replace(microsecond=0)
    logs = generate_logs(end=end)
    with open(LOG_FILE, "w") as f:
        for row in logs:
            f.write(json.dumps(row) + "\n")
    metrics = generate_metrics(end=end)
    metrics.to_csv(METRICS_FILE, index=False)
    print(f"Wrote {len(logs):,} log lines   -> {LOG_FILE}")
    print(f"Wrote {len(metrics):,} metric points -> {METRICS_FILE}")
    print("\nSample incident lines:")
    for row in [l for l in logs if l["level"] in ("ERROR", "CRITICAL")][-4:]:
        print(" ", row)
