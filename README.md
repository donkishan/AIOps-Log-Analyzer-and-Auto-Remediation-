# AIOps Log Analyzer & Auto-Remediation: live teaching guide (Windows)

An AI pipeline that **ingests server logs → clusters errors → detects metric anomalies and predicts crashes → asks an LLM for a fix (bash / kubectl / Terraform) → checks it with safety guardrails → waits for a human to approve it**. You can watch every stage in a browser dashboard.

```
 Server logs ─┐                          ┌─ Mask + TF-IDF ─ DBSCAN clusters ─┐
              ├─► DuckDB (+ Elasticsearch)┤                                   ├─► Correlate ─► LLM fix ─► Guardrails ─► Human approval
 Metrics ─────┘                          └─ Prophet forecast + crash ETA ────┘     (Ollama / Groq)
```

| Layer | Free tool used | Enterprise equivalent |
|---|---|---|
| Log store | DuckDB (embedded), optional Elasticsearch + Kibana in Docker | Splunk, Datadog |
| Log clustering | Regex masking + TF-IDF + DBSCAN (scikit-learn) | Splunk ITSI, Moogsoft |
| Metric anomalies | Prophet (falls back to a seasonal mean ± 4σ baseline) | Dynatrace Davis |
| Fix generation | Qwen2.5-Coder on Ollama (offline), or Groq's free tier | ServiceNow AIOps |
| UI | Streamlit | Custom NOC dashboard |

---

## 0. Before class (one-time installs)

1. **Python 3.10 – 3.12** from python.org. Tick **"Add python.exe to PATH"**.
2. **VS Code** (optional, good for showing files on a projector).
3. **LLM, pick one:**
   - **Ollama** (laptop with 8 GB+ RAM): install from https://ollama.com/download, then in PowerShell:
     ```powershell
     ollama pull qwen2.5-coder:7b      # ~4.7 GB. With 8 GB RAM use qwen2.5-coder:3b
     ollama run qwen2.5-coder:7b "write a bash command to restart a k8s deployment"
     ```
   - **Groq** (low-end hardware): create a free API key at https://console.groq.com/keys
   - **Neither**: set `LLM_PROVIDER=offline`. Built-in playbooks keep the demo working with no LLM.
4. **Docker Desktop** (only if you want to show Elasticsearch + Kibana).

> Tip: also run steps 1–9 once the night before. Then the downloads are cached and nothing new installs live.

---

## 1. Create the project (live)

```powershell
mkdir aiops-log-analyzer; cd aiops-log-analyzer
mkdir data, remediations, reports
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Create **`requirements.txt`**, then:
```powershell
pip install -r requirements.txt
```
(Or copy the whole folder and run `.\setup.ps1`.)

Create **`.env`** by copying `.env.example`, and set `LLM_PROVIDER=ollama`, `groq` or `offline`.

---

## 2. Build it file by file

Each step: **create the file → run it → explain the output**. Every module has a `__main__` block, so each one runs and shows output by itself.

| # | Create file | Run | What to explain |
|---|---|---|---|
| 1 | `config.py` | – | One place for settings; `.env` controls the LLM provider and the ROI numbers. |
| 2 | `generate_logs.py` | `python generate_logs.py` | We simulate 5 micro-services (48 h of logs, 7 days of metrics) and inject 4 real incidents: **OOMKilled pods, disk full, Redis timeouts, DB connection pool exhausted**. |
| 3 | `ingest.py` | `python ingest.py` | DuckDB reads JSON/CSV with **one SQL line**, with no server and no licence. Contrast this with Splunk's cost. Show the level counts and the last errors. |
| 4 | `clustering.py` | `python clustering.py` | **Masking**: `pod payment-api-3fa9c2d1` becomes `pod payment-api-<ID>`, so 14k lines turn into about 20 templates. **TF-IDF** turns text into vectors. **DBSCAN** groups them without needing *k*. **Spike ratio** is the recent rate ÷ the historical rate. **Correlation** merges the clusters of one service into a single incident. |
| 5 | `anomaly.py` | `python anomaly.py` | **Prophet** learns normal behaviour (trend + daily seasonality) from history that excludes the last 6 h, then flags points outside the 99% band. **Crash prediction**: a linear trend gives "memory hits 95% in ~15 min", so you can fix it *before* the outage. |
| 6 | `remediation.py` | `python remediation.py` | The prompt asks for strict JSON: diagnosis, script, rollback, verification and risk. **Guardrails** use regex to block `rm -rf /`, `terraform destroy`, `DROP DATABASE` and similar. Generated code is **never auto-executed**; it is saved to `remediations/` for a human to approve. |
| 7 | `pipeline.py` | `python pipeline.py` | The whole loop in one command, which writes `reports/report_*.json`. `python pipeline.py --watch 60` repeats it every minute. |
| 8 | `app.py` | `streamlit run app.py` | Opens **http://localhost:8501**, the browser UI (see section 4). |

Expected output of `python pipeline.py`:
```
[2] Log clusters      : 17 total, 8 anomalous -> 4 correlated incidents
[3] Metric anomalies  : 7 (prophet)
      payment-api    mem_pct    now=   91.6 expected=   58.2  -> hits 95 in ~14.7 min
      nginx-gateway  disk_pct   now=   97.9 expected=   63.0  -> ALREADY over limit
[4] INC-...-payment-api   -> bash      via ollama [PENDING_APPROVAL] remediations\INC-...-payment-api.sh
[4] INC-...-postgres-db   -> terraform via ollama [PENDING_APPROVAL] remediations\INC-...-postgres-db.tf
[5] ROI: ~160 min downtime saved (~$800,000)
```

---

## 3. Switching the LLM

In `.env`:
```
LLM_PROVIDER=ollama            # Ollama must be running (it starts automatically after install)
OLLAMA_MODEL=qwen2.5-coder:7b
# or
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
```
You can also switch providers live from the UI sidebar. If an LLM call fails (Ollama isn't running, no internet, bad key), the app falls back to the offline playbook and shows a warning, so the class never stalls.

Groq changes its model list over time. If a model name stops working, pick a current one from https://console.groq.com/docs/models.

---

## 4. Showing the workflow in the browser (demo script, ~15 min)

```powershell
streamlit run app.py
```

1. **Sidebar → "Generate data + ingest".** The workflow diagram at the top turns **green** stage by stage, and the KPI row fills in.
2. **Tab 1 · Ingest**: the log-volume chart has an obvious red spike at the end. Run a live SQL query in the DuckDB box, e.g. errors per service.
3. **Tab 2 · Cluster logs**: type any log line into the masking box to show the template. In the scatter plot, red dots are anomalous clusters. Move the **eps slider** in the sidebar and watch the clusters merge and split. Open the incident expanders to show 8 clusters becoming 4 incidents.
4. **Tab 3 · Forecast**: pick `payment-api / mem_pct`. Blue band = Prophet's expected range, red dots = anomalies, dashed line = limit. Point at **"Time to limit: ~15 min"**: *"we predicted the crash before it happened."*
5. **Tab 4 · Remediate**: pick the payment-api incident, open "Context sent to the LLM", then click **Generate fix**. Walk through the diagnosis, the kubectl script, the green guardrail badge, then **Approve → Simulate execution**. Repeat for postgres-db to show **Terraform** output.
6. **Guardrail playground** (bottom of tab 4): paste `rm -rf /` and show it get **blocked**. This is the "why AI needs guardrails in ops" moment.
7. **Tab 5 · ROI**: change the downtime cost or MTTR and see the dollars saved. Download the incident report.
8. **Live mode**: pick *Inject incident = redis* in the sidebar and turn on **Live mode**. New logs stream in every 5 s, and the counters and chart grow in real time.

**Two-terminal "continuous ingestion" demo** (the real-world pattern):
```powershell
# Terminal 1
streamlit run app.py
# Terminal 2 (activate the venv first)
python ingest.py --live --incident dbpool     # streams logs into DuckDB every 3 s
# or tail a real log file:
python ingest.py --follow C:\path\to\app.jsonl
```
Click **Stream 30s of logs now** or refresh the UI to see the new data.

---

## 5. Optional: Elasticsearch + Kibana

```powershell
docker compose up -d                 # wait ~60 s
# in .env:  ENABLE_ELASTICSEARCH=true
python ingest.py                     # logs now also go to index "server-logs"
```
Open **http://localhost:5601**, go to *Stack Management → Data Views → Create*, enter `server-logs` with time field `timestamp`, then use **Discover**. DuckDB remains the analytics engine; Elasticsearch only mirrors the logs for Kibana. If Elasticsearch is down, ingestion skips it and keeps going.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Prophet fails to install | Use Python 3.11. The app still works: choose **seasonal** in the sidebar. |
| `ollama failed (ConnectionError)` | Start Ollama from the Start menu, or run `ollama serve`. Check with `ollama list`. |
| Ollama is very slow | Use `qwen2.5-coder:3b` or `1.5b`, or switch to Groq. |
| `IO Error: could not set lock` | Two processes held DuckDB at the same moment. The code retries; if it persists, stop `ingest.py --live`. |
| Incidents disappear after an hour of class | The "recent" window moved past them. Click **Generate data + ingest** again. |
| Port 8501 is busy | `streamlit run app.py --server.port 8502` |

---

## 7. Extensions to set as assignments

- Replace the synthetic generator with real logs (`journalctl -o json`, nginx access logs, `kubectl logs`).
- Use the **Drain3** library for log template mining instead of regex masking.
- Send Slack / Teams alerts when an incident is created.
- Run approved scripts against a **kind/minikube** cluster or in a sandbox container.
- Add RAG over past incident runbooks so the LLM learns the team's own fixes.
- Track MTTR before and after across many runs to put a number on the ROI.
