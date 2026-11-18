"""Step 8 - Streamlit UI: shows the whole AIOps workflow in the browser.

Run:  streamlit run app.py      -> opens http://localhost:8501
"""
import json
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from anomaly import HAS_PROPHET, LIMITS, detect, predict_breach, scan_all
from clustering import cluster_logs, group_incidents, mask_message
from generate_logs import generate_live_batch
from ingest import (append_logs, get_logs, ingest_logs_file, ingest_metrics_file,
                    level_timeline, query, stats)
from remediation import build_incident, check_safety, generate_fix, save_fix

st.set_page_config(page_title="AIOps Log Analyzer", page_icon="🛠️", layout="wide")
ss = st.session_state
ss.setdefault("data_version", 0)    # bumps when logs change
ss.setdefault("metrics_version", 0) # bumps when metrics change
ss.setdefault("fixes", {})


# ------------------------------------------------------------------ cached computations
@st.cache_data(show_spinner=False)
def cached_clusters(version, eps, recent_minutes):
    errors = get_logs(levels=["WARN", "ERROR", "CRITICAL"])
    logs, summary, points = cluster_logs(errors, eps=eps, recent_minutes=recent_minutes)
    return summary, points


@st.cache_data(show_spinner=False)
def cached_scan(version, method):
    return scan_all(method)


@st.cache_data(show_spinner=False)
def cached_detect(version, service, metric, method):
    return detect(service, metric, method)


def bump():
    ss.data_version += 1


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Control panel")
    if st.button("1️⃣ Generate data + ingest", use_container_width=True, type="primary"):
        import subprocess, sys
        with st.spinner("Generating logs & metrics..."):
            subprocess.run([sys.executable, "generate_logs.py"], cwd=config.BASE_DIR, check=True)
            ingest_logs_file(); ingest_metrics_file()
        ss.fixes = {}; bump(); ss.metrics_version += 1; st.success("Data ready")

    st.divider()
    st.subheader("🔴 Live simulation")
    inc_choice = st.selectbox("Inject incident", ["none", "oom", "disk", "redis", "dbpool"])
    if st.button("Stream 30s of logs now", use_container_width=True):
        n = append_logs(generate_live_batch(30, None if inc_choice == "none" else inc_choice))
        bump(); st.toast(f"Ingested {n} new log lines")
    live = st.toggle("Live mode (auto-ingest every 5s)")

    st.divider()
    st.subheader("🧠 Models")
    provider = st.selectbox("LLM provider", ["ollama", "groq", "offline"],
                            index=["ollama", "groq", "offline"].index(config.LLM_PROVIDER)
                            if config.LLM_PROVIDER in ("ollama", "groq", "offline") else 2)
    st.caption({"ollama": f"Local model: `{config.OLLAMA_MODEL}`",
                "groq": f"Groq cloud: `{config.GROQ_MODEL}`",
                "offline": "Rule-based playbooks (no LLM)"}[provider])
    method = st.radio("Metric anomaly model", ["prophet", "seasonal"],
                      index=0 if HAS_PROPHET else 1, horizontal=True,
                      help="seasonal = mean ± 4σ per time-of-day slot (fallback if Prophet isn't installed)")
    eps = st.slider("DBSCAN eps (cluster tightness)", 0.1, 0.9, 0.5, 0.05)
    recent = st.slider("'Recent' window (minutes)", 15, 180, 60, 15)
    st.divider()
    st.caption(f"Store: DuckDB `{config.DUCKDB_PATH.split('/')[-1]}`"
               + (" + Elasticsearch" if config.ENABLE_ELASTICSEARCH else ""))

counts = stats()
if not counts:
    st.title("🛠️ AIOps Log Analyzer & Auto-Remediation")
    st.info("👈 Click **Generate data + ingest** in the sidebar to start.")
    st.stop()

# ------------------------------------------------------------------ run the pipeline
summary, points = cached_clusters(ss.data_version, eps, recent)
incidents = group_incidents(summary)
metric_anoms = cached_scan(ss.metrics_version, method)
anom_clusters = int(summary["anomalous"].sum()) if len(summary) else 0
approved = sum(1 for f in ss.fixes.values() if f["status"] == "APPROVED")

# ------------------------------------------------------------------ header + workflow diagram
st.title("🛠️ AIOps Log Analyzer & Auto-Remediation")
st.caption("Ingest → Cluster → Forecast → Correlate → Generate fix (LLM) → Guardrails → Human approval")


def node(name, label, done):
    color = "#2e7d32" if done else "#9e9e9e"
    return f'{name} [label="{label}", style="filled,rounded", shape=box, fillcolor="{color}", fontcolor=white, fontname=Helvetica];'


total_logs = sum(counts.values())
es_label = "\\n+ Elasticsearch" if config.ENABLE_ELASTICSEARCH else ""
n_generated = len(ss.fixes)
NL = "\\n"  # graphviz line break
nodes = [
    node("logs", f"Server logs{NL}{total_logs:,} lines", True),
    node("metrics", f"Server metrics{NL}CPU/Mem/Disk/Latency", True),
    node("ingest", "DuckDB ingest" + es_label, True),
    node("mask", f"Mask + TF-IDF{NL}(log templates)", True),
    node("cluster", f"DBSCAN{NL}{len(summary)} clusters{NL}{anom_clusters} anomalous", len(summary) > 0),
    node("prophet", f"{method.title()} forecast{NL}{len(metric_anoms)} metric anomalies", len(metric_anoms) > 0),
    node("corr", f"Correlate{NL}{len(incidents)} incidents", len(incidents) > 0),
    node("llm", f"LLM fix ({provider}){NL}{n_generated} generated", n_generated > 0),
    node("guard", f"Guardrails{NL}(safety check)", n_generated > 0),
    node("human", f"Human approval{NL}{approved} approved", approved > 0),
]
dot = ("digraph G { rankdir=LR; bgcolor=transparent; edge [color=\"#888\"];\n" + "\n".join(nodes) +
       "\nlogs -> ingest; metrics -> ingest; ingest -> mask -> cluster -> corr;"
       "\ningest -> prophet -> corr; corr -> llm -> guard -> human; }")
st.graphviz_chart(dot, use_container_width=True)

saved_min = approved * (config.MANUAL_MTTR_MIN - config.AUTO_MTTR_MIN)
k = st.columns(6)
k[0].metric("Log lines", f"{total_logs:,}")
k[1].metric("Errors (ERROR+CRIT)", f"{counts.get('ERROR', 0) + counts.get('CRITICAL', 0):,}")
k[2].metric("Anomalous clusters", anom_clusters)
k[3].metric("Metric anomalies", len(metric_anoms))
k[4].metric("Incidents", len(incidents))
k[5].metric("Downtime saved", f"{saved_min:.0f} min", f"${saved_min * config.DOWNTIME_COST_PER_MIN:,.0f}")

t1, t2, t3, t4, t5 = st.tabs(["1 · Ingest", "2 · Cluster logs", "3 · Forecast metrics",
                              "4 · Remediate", "5 · ROI & report"])

# ------------------------------------------------------------------ TAB 1: ingest
with t1:
    st.subheader("Log volume by level")
    tl = level_timeline(15)
    fig = px.area(tl, x="bucket", y="n", color="level",
                  color_discrete_map={"INFO": "#90caf9", "WARN": "#ffb74d", "ERROR": "#e57373", "CRITICAL": "#b71c1c"})
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), xaxis_title=None, yaxis_title="lines / 15 min")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Latest errors")
        st.dataframe(get_logs(["ERROR", "CRITICAL"], limit=200), use_container_width=True, height=320, hide_index=True)
    with c2:
        st.subheader("Query logs with SQL (DuckDB)")
        sql = st.text_area("SQL", "SELECT service, level, COUNT(*) AS n\nFROM logs\nWHERE level IN ('ERROR','CRITICAL')\nGROUP BY 1, 2 ORDER BY n DESC", height=140)
        if st.button("Run SQL"):
            try:
                st.dataframe(query(sql), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(str(e))
        if config.ENABLE_ELASTICSEARCH:
            st.link_button("Open Kibana ↗", "http://localhost:5601")

# ------------------------------------------------------------------ TAB 2: cluster
with t2:
    st.subheader("Step A · Masking turns messages into templates")
    raw = st.text_input("Try any log line", "OOMKilled: container payment-api in pod payment-api-3fa9c2d1 exceeded memory limit 512Mi")
    st.code(mask_message(raw), language="text")

    st.subheader("Step B · TF-IDF + DBSCAN clusters (each dot = one template, size = count)")
    if len(points):
        pts = points.copy()
        pts["status"] = pts["anomalous"].map({True: "anomalous", False: "normal"}).fillna("normal")
        pts.loc[pts["cluster"] == -1, "status"] = "rare / noise"
        fig = px.scatter(pts, x="x", y="y", size="count", color="status", hover_data=["template", "cluster", "count"],
                         color_discrete_map={"anomalous": "#d32f2f", "normal": "#1976d2", "rare / noise": "#9e9e9e"},
                         size_max=45)
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="SVD-1", yaxis_title="SVD-2")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Step C · Cluster table (spike = recent rate ÷ historical rate)")
    st.dataframe(summary[["cluster", "anomalous", "level", "count", "recent_count", "spike_ratio", "is_new", "services", "label"]],
                 use_container_width=True, hide_index=True,
                 column_config={"spike_ratio": st.column_config.ProgressColumn("spike_ratio", max_value=float(summary["spike_ratio"].max() or 1), format="%.0fx")})

    st.subheader(f"Step D · Correlated incidents ({len(incidents)})")
    for i, inc in enumerate(incidents):
        with st.expander(f"🚨 [{inc['level']}] {inc['services']} — {inc['recent_count']} errors in last {recent} min, clusters {inc['clusters']}"):
            for t in inc["templates"]:
                st.code(t, language="text")

# ------------------------------------------------------------------ TAB 3: forecast
with t3:
    st.subheader("Services misbehaving right now")
    if len(metric_anoms):
        ma = metric_anoms.copy()
        ma["status"] = ma["eta_to_limit_min"].apply(
            lambda e: "🔥 over limit" if e == 0 else (f"⏱️ breach in ~{e:.0f} min" if pd.notna(e) and e else "⚠️ anomalous"))
        st.dataframe(ma, use_container_width=True, hide_index=True)
    else:
        st.success("No metric anomalies.")

    c1, c2 = st.columns(2)
    svc = c1.selectbox("Service", config.SERVICES)
    met = c2.selectbox("Metric", config.METRICS, index=1)
    with st.spinner(f"Fitting {method} model..."):
        res = cached_detect(ss.metrics_version, svc, met, method)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["ds"], y=res["yhat_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=res["ds"], y=res["yhat_lower"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(25,118,210,0.15)", name="expected range"))
    fig.add_trace(go.Scatter(x=res["ds"], y=res["yhat"], line=dict(color="#1976d2", dash="dot"), name="forecast"))
    fig.add_trace(go.Scatter(x=res["ds"], y=res["y"], line=dict(color="#424242", width=1), name="actual"))
    an = res[res["anomaly"]]
    fig.add_trace(go.Scatter(x=an["ds"], y=an["y"], mode="markers", marker=dict(color="#d32f2f", size=6), name="anomaly"))
    fig.add_hline(y=LIMITS[met], line_dash="dash", line_color="#d32f2f", annotation_text="limit")
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    b = predict_breach(svc, met)
    if b:
        cols = st.columns(3)
        cols[0].metric("Current", f"{b['current']:.1f}")
        cols[1].metric("Trend / min", f"{b['slope_per_min']:+.2f}")
        cols[2].metric("Time to limit", "already breached" if b["eta_min"] == 0 else
                       (f"{b['eta_min']:.0f} min" if b["eta_min"] else "not trending up"))

# ------------------------------------------------------------------ TAB 4: remediate
with t4:
    if not incidents:
        st.success("No incidents to remediate 🎉")
    else:
        labels = [f"[{i['level']}] {i['services']} ({i['recent_count']} errors)" for i in incidents]
        pick = st.selectbox("Choose incident", range(len(incidents)), format_func=lambda i: labels[i])
        incident = build_incident(incidents[pick], metric_anoms)
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("**Context sent to the LLM**")
            st.json(incident, expanded=False)
            if st.button(f"🤖 Generate fix with {provider}", type="primary", use_container_width=True):
                with st.spinner(f"Asking {provider}..."):
                    ss.fixes[incident["id"]] = generate_fix(incident, provider)
                st.rerun()  # refresh the workflow diagram at the top
            if st.button("⚡ Generate fixes for ALL incidents", use_container_width=True):
                prog = st.progress(0.0)
                for n, g in enumerate(incidents):
                    inc = build_incident(g, metric_anoms)
                    ss.fixes[inc["id"]] = generate_fix(inc, provider)
                    prog.progress((n + 1) / len(incidents))
                st.rerun()
        with c2:
            fix = ss.fixes.get(incident["id"])
            if not fix:
                st.info("Click **Generate fix** to call the LLM.")
            else:
                if fix["note"]:
                    st.warning(fix["note"])
                st.markdown(f"**Diagnosis:** {fix['diagnosis']}  \n**Root cause:** {fix['root_cause']}  \n"
                            f"**Risk:** `{fix['risk']}` · **Model:** `{fix['model']}` · **Latency:** {fix['latency_s']}s")
                st.code(fix["script"], language="hcl" if fix["script_type"] == "terraform" else "bash")
                if fix["safety_issues"]:
                    st.error("🛑 Guardrails BLOCKED this script: " + "; ".join(fix["safety_issues"]))
                else:
                    st.success("✅ Guardrails passed — waiting for human approval")
                with st.expander("Rollback & verification"):
                    st.markdown(f"**Rollback:** `{fix['rollback']}`")
                    st.markdown(f"**Verify:** `{fix['verification']}`")
                a1, a2, a3 = st.columns(3)
                if a1.button("👍 Approve", disabled=bool(fix["safety_issues"]), use_container_width=True):
                    fix["status"] = "APPROVED"; fix["approved_at"] = datetime.now().isoformat()
                    st.toast(f"Saved to {save_fix(fix)}"); st.rerun()
                if a2.button("👎 Reject", use_container_width=True):
                    fix["status"] = "REJECTED"; st.rerun()
                a3.download_button("⬇️ Download", fix["script"], use_container_width=True,
                                   file_name=f"{fix['incident_id']}.{'tf' if fix['script_type'] == 'terraform' else 'sh'}")
                st.markdown(f"**Status:** `{fix['status']}`")
                if fix["status"] == "APPROVED" and st.button("▶️ Simulate execution (dry run)"):
                    box = st.empty(); out = ""
                    steps = [l for l in fix["script"].splitlines() if l.strip() and not l.strip().startswith(("#", "set "))]
                    for line in steps[:12]:
                        out += f"$ {line.strip()}\n  ✓ ok (dry-run)\n"
                        box.code(out, language="bash"); time.sleep(0.3)
                    st.success("Dry run finished. In production this would go to Ansible / ArgoCD / a CI job.")

    st.divider()
    st.subheader("🧪 Guardrail playground — what if the LLM hallucinates something dangerous?")
    bad = st.text_area("Paste any script", "kubectl rollout restart deployment/payment-api\nrm -rf / --no-preserve-root")
    issues = check_safety(bad)
    (st.error("Blocked: " + "; ".join(issues)) if issues else st.success("Looks safe"))

# ------------------------------------------------------------------ TAB 5: ROI
with t5:
    st.subheader("Return on investment")
    c1, c2, c3 = st.columns(3)
    cost = c1.number_input("Downtime cost ($ / min)", value=config.DOWNTIME_COST_PER_MIN, step=500.0)
    manual = c2.number_input("Manual MTTR (min)", value=config.MANUAL_MTTR_MIN)
    auto = c3.number_input("AIOps MTTR (min)", value=config.AUTO_MTTR_MIN)
    n_inc = max(len(incidents), 1)
    roi = pd.DataFrame({"approach": ["Manual on-call", "AIOps"], "minutes of downtime": [manual * n_inc, auto * n_inc]})
    fig = px.bar(roi, x="approach", y="minutes of downtime", color="approach", text_auto=True,
                 color_discrete_sequence=["#9e9e9e", "#2e7d32"])
    fig.update_layout(height=320, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.metric(f"Saved across {n_inc} incidents", f"${(manual - auto) * n_inc * cost:,.0f}",
              f"{(manual - auto) * n_inc:.0f} minutes less downtime")
    report = {"generated_at": datetime.now().isoformat(), "incidents": incidents,
              "metric_anomalies": metric_anoms.to_dict("records"), "fixes": list(ss.fixes.values())}
    st.download_button("⬇️ Download incident report (JSON)", json.dumps(report, indent=2, default=str),
                       file_name=f"aiops_report_{datetime.now():%Y%m%d_%H%M}.json")

# ------------------------------------------------------------------ live mode loop
if live:
    time.sleep(5)
    append_logs(generate_live_batch(5, None if inc_choice == "none" else inc_choice))
    bump()
    st.rerun()
