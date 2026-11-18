"""Step 1 - Central configuration. Every other file imports from here."""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
REMEDIATION_DIR = BASE_DIR / "remediations"
REPORT_DIR = BASE_DIR / "reports"
for d in (DATA_DIR, REMEDIATION_DIR, REPORT_DIR):
    d.mkdir(exist_ok=True)

LOG_FILE = DATA_DIR / "app_logs.jsonl"
METRICS_FILE = DATA_DIR / "metrics.csv"
DUCKDB_PATH = str(BASE_DIR / os.getenv("DUCKDB_PATH", "data/aiops.duckdb"))

# LLM
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Elasticsearch (optional)
ENABLE_ELASTICSEARCH = os.getenv("ENABLE_ELASTICSEARCH", "false").lower() == "true"
ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "server-logs")

# ROI
DOWNTIME_COST_PER_MIN = float(os.getenv("DOWNTIME_COST_PER_MIN", 5000))
MANUAL_MTTR_MIN = float(os.getenv("MANUAL_MTTR_MIN", 45))
AUTO_MTTR_MIN = float(os.getenv("AUTO_MTTR_MIN", 5))

SERVICES = ["payment-api", "auth-service", "nginx-gateway", "redis-cache", "postgres-db"]
METRICS = ["cpu_pct", "mem_pct", "disk_pct", "latency_ms"]
