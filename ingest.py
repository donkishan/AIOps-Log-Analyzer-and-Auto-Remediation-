"""Step 3 - Log ingestion into DuckDB (+ optional mirror to Elasticsearch).

DuckDB = a free, embedded analytics database (think "SQLite for analytics").
No server, no Docker - perfect replacement for Splunk in a demo.

Run:
  python ingest.py                 # load data/app_logs.jsonl + data/metrics.csv
  python ingest.py --live          # keep streaming fresh logs every 3s (Ctrl+C to stop)
  python ingest.py --live --incident redis   # stream with an incident injected
  python ingest.py --follow path/to/file.jsonl   # tail a real JSONL log file
"""
import argparse
import json
import time

import duckdb
import pandas as pd

import config


# ------------------------------------------------------------------ helpers
def _connect(read_only: bool = False, retries: int = 20):
    """Short-lived connections + retry so the UI and the live streamer can share the file."""
    for _ in range(retries):
        try:
            return duckdb.connect(config.DUCKDB_PATH, read_only=read_only)
        except duckdb.IOException:
            time.sleep(0.25)
    return duckdb.connect(config.DUCKDB_PATH, read_only=read_only)


def query(sql: str, params: list | None = None) -> pd.DataFrame:
    con = _connect()
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def init_db():
    con = _connect()
    con.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            timestamp TIMESTAMP, service VARCHAR, host VARCHAR,
            level VARCHAR, message VARCHAR)""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            timestamp TIMESTAMP, service VARCHAR, metric VARCHAR, value DOUBLE)""")
    con.close()


# ------------------------------------------------------------------ ingestion
def ingest_logs_file(path=config.LOG_FILE) -> int:
    init_db()
    con = _connect()
    con.execute("DELETE FROM logs")
    # DuckDB reads JSON-lines natively - one line of SQL!
    con.execute(f"INSERT INTO logs SELECT CAST(timestamp AS TIMESTAMP), service, host, level, message "
                f"FROM read_json_auto('{str(path).replace(chr(92), '/')}')")
    n = con.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
    con.close()
    if config.ENABLE_ELASTICSEARCH:
        with open(path) as f:
            ship_to_elasticsearch([json.loads(l) for l in f])
    return n


def ingest_metrics_file(path=config.METRICS_FILE) -> int:
    init_db()
    con = _connect()
    con.execute("DELETE FROM metrics")
    con.execute(f"INSERT INTO metrics SELECT CAST(timestamp AS TIMESTAMP), service, metric, value "
                f"FROM read_csv_auto('{str(path).replace(chr(92), '/')}')")
    n = con.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    con.close()
    return n


def append_logs(records: list[dict]) -> int:
    if not records:
        return 0
    init_db()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    con = _connect()
    con.register("batch", df)
    con.execute("INSERT INTO logs SELECT timestamp, service, host, level, message FROM batch")
    con.close()
    if config.ENABLE_ELASTICSEARCH:
        ship_to_elasticsearch(records)
    return len(records)


def ship_to_elasticsearch(records: list[dict]):
    """Optional: bulk-index logs so they can be explored in Kibana."""
    try:
        from elasticsearch import Elasticsearch, helpers
        es = Elasticsearch(config.ES_URL)
        helpers.bulk(es, ({"_index": config.ES_INDEX, "_source": r} for r in records))
        print(f"  [ES] indexed {len(records)} docs into '{config.ES_INDEX}'")
    except Exception as e:  # never break the pipeline because ES is down
        print(f"  [ES] skipped ({e.__class__.__name__}: {e})")


# ------------------------------------------------------------------ queries used by the UI
def get_logs(levels=None, minutes: int | None = None, limit: int | None = None) -> pd.DataFrame:
    sql, params = "SELECT * FROM logs WHERE 1=1", []
    if levels:
        sql += f" AND level IN ({','.join('?' * len(levels))})"
        params += list(levels)
    if minutes:
        sql += " AND timestamp >= (SELECT MAX(timestamp) FROM logs) - INTERVAL (?) MINUTE"
        params.append(minutes)
    sql += " ORDER BY timestamp DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return query(sql, params)


def level_timeline(bucket_minutes: int = 15) -> pd.DataFrame:
    return query(f"""
        SELECT time_bucket(INTERVAL '{bucket_minutes} minutes', timestamp) AS bucket, level, COUNT(*) AS n
        FROM logs GROUP BY 1, 2 ORDER BY 1""")


def get_metrics(service=None, metric=None) -> pd.DataFrame:
    sql, params = "SELECT * FROM metrics WHERE 1=1", []
    if service:
        sql += " AND service = ?"; params.append(service)
    if metric:
        sql += " AND metric = ?"; params.append(metric)
    return query(sql + " ORDER BY timestamp", params)


def stats() -> dict:
    init_db()
    df = query("SELECT level, COUNT(*) n FROM logs GROUP BY level")
    return dict(zip(df.level, df.n))


# ------------------------------------------------------------------ continuous modes
def live_stream(incident=None, interval=3):
    from generate_logs import generate_live_batch
    print(f"Streaming live logs every {interval}s (incident={incident}). Ctrl+C to stop.")
    while True:
        n = append_logs(generate_live_batch(seconds=interval, incident=incident))
        print(f"  +{n} lines  | totals: {stats()}")
        time.sleep(interval)


def follow_file(path, interval=2):
    print(f"Tailing {path} ... Ctrl+C to stop.")
    with open(path) as f:
        f.seek(0, 2)  # jump to end, like `tail -f`
        while True:
            lines = [json.loads(l) for l in f.readlines() if l.strip()]
            if lines:
                print(f"  +{append_logs(lines)} lines")
            time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--incident", choices=["oom", "disk", "redis", "dbpool"])
    ap.add_argument("--follow")
    a = ap.parse_args()
    if a.live:
        live_stream(a.incident)
    elif a.follow:
        follow_file(a.follow)
    else:
        print(f"Logs ingested   : {ingest_logs_file():,}")
        print(f"Metrics ingested: {ingest_metrics_file():,}")
        print("Level counts    :", stats())
        print("\nLast 5 errors:")
        print(get_logs(["ERROR", "CRITICAL"], limit=5)[["timestamp", "service", "message"]].to_string(index=False))
