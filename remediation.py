"""Step 6 - LLM-generated remediation (shell / kubectl / Terraform) with safety guardrails.

Providers (set LLM_PROVIDER in .env):
  ollama  -> local Qwen2.5-Coder via Ollama (offline, free)
  groq    -> Groq free tier (cloud, very fast)
  offline -> built-in playbooks (no LLM; used automatically if the LLM call fails)

Generated code is NEVER executed automatically - it is checked by guardrails,
saved to remediations/ and needs human approval (human-in-the-loop).

Run:  python remediation.py
"""
import json
import re
import time
from datetime import datetime

import requests

import config

SYSTEM_PROMPT = """You are a senior Site Reliability Engineer (SRE) for a Kubernetes-based
production platform running on AWS. Given an incident, produce a SAFE, minimal, idempotent fix.

Respond with ONLY a JSON object with exactly these keys:
{
  "diagnosis": "1-2 sentence explanation of what is happening",
  "root_cause": "most likely root cause",
  "script_type": "bash" | "terraform",
  "script": "the full remediation script (bash with kubectl/systemctl/redis-cli etc, or Terraform HCL)",
  "rollback": "how to undo the change",
  "verification": "commands to verify the fix worked",
  "risk": "low" | "medium" | "high"
}
Rules: bash scripts start with `set -euo pipefail` and use comments; prefer restarts, cache
flushes, log cleanup and config scaling over destructive actions; never delete data volumes,
databases or namespaces; use Terraform only for infrastructure/config changes."""

# ------------------------------------------------------------------ guardrails
DANGEROUS = [
    (r"rm\s+-rf\s+/(\s|$|\*)", "Deletes the root filesystem"),
    (r"\bmkfs\b", "Formats a disk"),
    (r"\bdd\s+if=", "Raw disk write"),
    (r":\(\)\s*\{", "Fork bomb"),
    (r"(^|[;&|]\s*|sudo\s+)(shutdown|reboot|halt|poweroff)\b", "Shuts down the host"),
    (r"drop\s+(database|table)", "Drops database objects"),
    (r"terraform\s+destroy", "Destroys infrastructure"),
    (r"kubectl\s+delete\s+(namespace|ns|pv|pvc)\b", "Deletes namespaces/volumes"),
    (r"chmod\s+(-R\s+)?777", "Opens permissions to everyone"),
    (r"curl[^|\n]*\|\s*(sudo\s+)?(ba)?sh", "Pipes internet script into shell"),
]


def check_safety(script: str) -> list[str]:
    return [reason for pattern, reason in DANGEROUS if re.search(pattern, script, re.IGNORECASE | re.MULTILINE)]


# ------------------------------------------------------------------ incident context
def build_incident(group: dict, metric_anomalies=None) -> dict:
    """group = one correlated incident from clustering.group_incidents()."""
    svcs = [s.strip() for s in group["services"].split(",")]
    related = []
    if metric_anomalies is not None and len(metric_anomalies):
        related = metric_anomalies[metric_anomalies["service"].isin(svcs)].to_dict("records")
    return {
        "id": f"INC-{datetime.now():%Y%m%d}-{svcs[0]}",
        "services": group["services"],
        "hosts": group["hosts"],
        "level": group["level"],
        "error_templates": group["templates"],
        "errors_last_hour": int(group["recent_count"]),
        "spike_ratio": group["spike_ratio"],
        "sample_logs": group["samples"],
        "related_metric_anomalies": related,
    }


def _user_prompt(incident: dict) -> str:
    return "Incident report:\n" + json.dumps(incident, indent=2, default=str)


# ------------------------------------------------------------------ providers
def _parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate ```json fences / extra prose
    return json.loads(m.group(0) if m else text)


def call_ollama(incident: dict) -> dict:
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", timeout=180, json={
        "model": config.OLLAMA_MODEL, "stream": False, "format": "json",
        "options": {"temperature": 0.1},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": _user_prompt(incident)}],
    })
    r.raise_for_status()
    return _parse_json(r.json()["message"]["content"])


def call_groq(incident: dict) -> dict:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is empty in .env")
    r = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=60,
                      headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"}, json={
        "model": config.GROQ_MODEL, "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": _user_prompt(incident)}],
    })
    r.raise_for_status()
    return _parse_json(r.json()["choices"][0]["message"]["content"])


PLAYBOOKS = [
    (r"OOMKilled|OutOfMemory|Back-off restarting", {
        "diagnosis": "payment-api pods are exceeding their 512Mi memory limit and being OOMKilled, causing a restart loop.",
        "root_cause": "Memory leak in the payment processor or an under-sized memory limit.",
        "script_type": "bash",
        "script": """#!/usr/bin/env bash
set -euo pipefail
NS=production
DEPLOY=payment-api

# 1. Capture evidence before changing anything
kubectl -n "$NS" get pods -l app=$DEPLOY -o wide
kubectl -n "$NS" top pods -l app=$DEPLOY || true

# 2. Temporarily raise the memory limit to stop the crash loop
kubectl -n "$NS" set resources deployment/$DEPLOY --limits=memory=1Gi --requests=memory=512Mi

# 3. Rolling restart to clear leaked memory (zero downtime)
kubectl -n "$NS" rollout restart deployment/$DEPLOY
kubectl -n "$NS" rollout status deployment/$DEPLOY --timeout=180s""",
        "rollback": "kubectl -n production rollout undo deployment/payment-api",
        "verification": "kubectl -n production get pods -l app=payment-api && kubectl -n production top pods -l app=payment-api",
        "risk": "low"}),
    (r"No space left|rotate log", {
        "diagnosis": "The nginx-gateway node disk is ~98% full; nginx can no longer write logs.",
        "root_cause": "Log rotation is failing so access/error logs grew until the disk filled.",
        "script_type": "bash",
        "script": """#!/usr/bin/env bash
set -euo pipefail
LOG_DIR=/var/log/nginx

df -h /var/log
# 1. Compress logs older than 1 day, delete compressed logs older than 7 days
find "$LOG_DIR" -name "*.log.*" ! -name "*.gz" -mtime +1 -exec gzip -f {} \\;
find "$LOG_DIR" -name "*.gz" -mtime +7 -delete

# 2. Truncate (not delete) the live logs so nginx keeps its file handles
truncate -s 0 "$LOG_DIR/access.log" "$LOG_DIR/error.log"

# 3. Force logrotate + reopen log files
logrotate -f /etc/logrotate.d/nginx
nginx -s reopen
df -h /var/log""",
        "rollback": "No rollback needed - only rotated/old logs were compressed or removed.",
        "verification": "df -h /var/log && tail -n 5 /var/log/nginx/error.log",
        "risk": "low"}),
    (r"Redis|evicting", {
        "diagnosis": "Redis has hit maxmemory and is timing out / evicting keys, slowing every service that uses the cache.",
        "root_cause": "Cache keys without TTL filled memory; eviction policy is not set to allkeys-lru.",
        "script_type": "bash",
        "script": """#!/usr/bin/env bash
set -euo pipefail
REDIS="redis-cli -h redis-cache -p 6379"

$REDIS INFO memory | grep -E "used_memory_human|maxmemory_human"
# 1. Evict least-recently-used keys instead of rejecting writes
$REDIS CONFIG SET maxmemory-policy allkeys-lru
# 2. Clear only the session cache namespace (safe to rebuild)
$REDIS --scan --pattern 'session:*' | xargs -r -n 500 $REDIS UNLINK
# 3. Restart the cache pods if latency is still high
kubectl -n production rollout restart statefulset/redis-cache""",
        "rollback": "redis-cli -h redis-cache CONFIG SET maxmemory-policy noeviction",
        "verification": "redis-cli -h redis-cache --latency -i 1 (Ctrl+C after 10s)",
        "risk": "medium"}),
    (r"Connection pool exhausted|too many clients", {
        "diagnosis": "Postgres has reached max_connections=100; services cannot get DB connections.",
        "root_cause": "Connection limit too low for current traffic and no pooler in front of the DB.",
        "script_type": "terraform",
        "script": """# Raise max_connections on the RDS Postgres instance
resource "aws_db_parameter_group" "postgres_primary" {
  name   = "postgres-primary-params"
  family = "postgres15"

  parameter {
    name         = "max_connections"
    value        = "300"
    apply_method = "pending-reboot"
  }
  parameter {
    name  = "idle_in_transaction_session_timeout"
    value = "60000" # kill idle-in-transaction sessions after 60s
  }
}

resource "aws_db_instance" "postgres_primary" {
  identifier           = "postgres-primary"
  parameter_group_name = aws_db_parameter_group.postgres_primary.name
  apply_immediately    = true
  # ... existing settings unchanged ...
}""",
        "rollback": "git revert the change and run terraform apply (max_connections back to 100).",
        "verification": "psql -c \"SHOW max_connections;\" && psql -c \"SELECT count(*) FROM pg_stat_activity;\"",
        "risk": "medium"}),
]


def offline_playbook(incident: dict) -> dict:
    text = " ".join(incident["error_templates"] + incident["sample_logs"])
    for pattern, fix in PLAYBOOKS:
        if re.search(pattern, text, re.IGNORECASE):
            return dict(fix)
    return {"diagnosis": "Unknown error pattern.", "root_cause": "Needs investigation.",
            "script_type": "bash", "risk": "low",
            "script": f"#!/usr/bin/env bash\nset -euo pipefail\n# Collect diagnostics\nkubectl -n production get pods\nkubectl -n production logs -l app={incident['services'].split(',')[0]} --tail=200",
            "rollback": "Nothing to roll back (read-only).", "verification": "Review collected logs."}


# ------------------------------------------------------------------ main entry
def generate_fix(incident: dict, provider: str | None = None) -> dict:
    provider = (provider or config.LLM_PROVIDER).lower()
    t0, note = time.time(), ""
    try:
        if provider == "ollama":
            fix, model = call_ollama(incident), config.OLLAMA_MODEL
        elif provider == "groq":
            fix, model = call_groq(incident), config.GROQ_MODEL
        else:
            fix, model = offline_playbook(incident), "rule-based playbook"
    except Exception as e:
        note = f"{provider} failed ({e.__class__.__name__}: {str(e)[:120]}) -> used offline playbook"
        fix, model, provider = offline_playbook(incident), "rule-based playbook", "offline"

    for k in ("diagnosis", "root_cause", "script_type", "script", "rollback", "verification", "risk"):
        fix.setdefault(k, "")
    fix["script_type"] = "terraform" if "terraform" in str(fix["script_type"]).lower() else "bash"
    fix.update({
        "incident_id": incident["id"], "provider": provider, "model": model,
        "latency_s": round(time.time() - t0, 2), "note": note,
        "safety_issues": check_safety(fix["script"]),
    })
    fix["status"] = "BLOCKED" if fix["safety_issues"] else "PENDING_APPROVAL"
    return fix


def save_fix(fix: dict) -> str:
    ext = "tf" if fix["script_type"] == "terraform" else "sh"
    path = config.REMEDIATION_DIR / f"{fix['incident_id']}.{ext}"
    path.write_text(fix["script"], encoding="utf-8")
    (config.REMEDIATION_DIR / f"{fix['incident_id']}.json").write_text(
        json.dumps(fix, indent=2, default=str), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    from clustering import detect_log_anomalies, group_incidents
    _, summary, _ = detect_log_anomalies()
    inc = build_incident(group_incidents(summary)[0])
    print("Incident:", json.dumps({k: inc[k] for k in ("id", "services", "error_templates")}, indent=2))
    fix = generate_fix(inc)
    print(f"\nProvider: {fix['provider']} ({fix['model']}) in {fix['latency_s']}s {fix['note']}")
    print("Diagnosis:", fix["diagnosis"])
    print(f"--- {fix['script_type']} ---\n{fix['script']}\n---")
    print("Guardrails:", fix["safety_issues"] or "passed", "| status:", fix["status"])
    print("Saved to:", save_fix(fix))
