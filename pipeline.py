"""Step 7 - The whole AIOps loop end-to-end from the command line.

  Ingest -> Cluster logs -> Metric anomalies + crash ETA -> Correlate -> LLM fix -> Guardrails -> Report

Run:  python pipeline.py                 (one pass)
      python pipeline.py --watch 60      (re-run every 60 seconds = continuous AIOps)
"""
import argparse
import json
import time
from datetime import datetime

import config
from anomaly import HAS_PROPHET, scan_all
from clustering import detect_log_anomalies, group_incidents
from ingest import ingest_logs_file, ingest_metrics_file, stats
from remediation import build_incident, generate_fix, save_fix


def run_once(reload_files: bool = True, provider: str | None = None) -> dict:
    t0 = time.time()
    print("=" * 70, f"\nAIOps run @ {datetime.now():%H:%M:%S}")
    if reload_files:
        if not config.LOG_FILE.exists():
            import subprocess, sys
            subprocess.run([sys.executable, "generate_logs.py"], check=True, cwd=config.BASE_DIR)
        ingest_logs_file(); ingest_metrics_file()
    print("[1] Ingested          :", stats())

    _, clusters, _ = detect_log_anomalies()
    anomalous = clusters[clusters["anomalous"]]
    incidents = group_incidents(clusters)
    print(f"[2] Log clusters      : {len(clusters)} total, {len(anomalous)} anomalous "
          f"-> {len(incidents)} correlated incidents")

    method = "prophet" if HAS_PROPHET else "seasonal"
    metric_anoms = scan_all(method)
    print(f"[3] Metric anomalies  : {len(metric_anoms)} ({method})")
    for r in metric_anoms.to_dict("records"):
        eta = r["eta_to_limit_min"]
        print(f"      {r['service']:14} {r['metric']:10} now={r['current']:>7} expected={r['expected']:>7}"
              + ("  -> ALREADY over limit" if eta == 0 else f"  -> hits {r['limit']} in ~{eta} min" if eta else ""))

    fixes = []
    for group in incidents:
        inc = build_incident(group, metric_anoms)
        fix = generate_fix(inc, provider)
        path = save_fix(fix)
        fixes.append({"incident": inc, "fix": fix, "file": path})
        print(f"[4] {inc['id']} {inc['services']:14} -> {fix['script_type']:9} via {fix['provider']:7} "
              f"[{fix['status']}] {path}")

    saved = len(fixes) * (config.MANUAL_MTTR_MIN - config.AUTO_MTTR_MIN)
    report = {
        "run_at": datetime.now().isoformat(), "duration_s": round(time.time() - t0, 1),
        "anomalous_clusters": len(anomalous), "incident_count": len(incidents), "metric_anomalies": metric_anoms.to_dict("records"),
        "incidents": fixes, "downtime_minutes_saved": saved,
        "cost_saved": saved * config.DOWNTIME_COST_PER_MIN,
    }
    out = config.REPORT_DIR / f"report_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[5] ROI: ~{saved:.0f} min downtime saved (~${report['cost_saved']:,.0f}). Report: {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, help="seconds between runs")
    ap.add_argument("--provider", choices=["ollama", "groq", "offline"])
    a = ap.parse_args()
    if a.watch:
        first = True
        while True:
            run_once(reload_files=first, provider=a.provider)
            first = False
            time.sleep(a.watch)
    else:
        run_once(provider=a.provider)
