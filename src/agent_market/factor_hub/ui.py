"""Streamlit dashboard for Factor Hub.

Run:
    streamlit run src/agent_market/factor_hub/ui.py
or via factor_lab CLI:
    python scripts/factor_lab.py hub ui
"""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List

try:
    import streamlit as st
except Exception as exc:  # noqa: BLE001
    raise SystemExit("Streamlit required: pip install streamlit") from exc

import pandas as pd

from .client import Client


# ============================================================
# Page config & cached client
# ============================================================

st.set_page_config(page_title="Factor Hub", page_icon="🧪", layout="wide")


@st.cache_resource(show_spinner=False)
def _client() -> Client:
    db_path = os.environ.get("FACTOR_HUB_DB") or None
    c = Client(db_path=db_path)
    c.init_db()
    return c


# ============================================================
# Helpers
# ============================================================

def _attach_metric(client: Client, factors: List[Dict], metric: str) -> pd.DataFrame:
    df = pd.DataFrame(factors)
    if df.empty:
        return df
    if "latest_metric" in df.columns and df["latest_metric"].notna().any():
        df = df.rename(columns={"latest_metric": metric})
    else:
        df[metric] = [client.latest_metric(int(fid), metric) for fid in df["id"]]
    keep = [c for c in ["id", "name", "category", "origin", "status",
                        metric, "complexity", "features_used", "expression",
                        "source_lib", "updated_at"] if c in df.columns]
    return df[keep]


# ============================================================
# Sidebar navigation
# ============================================================

client = _client()

st.sidebar.title("🧪 Factor Hub")
st.sidebar.caption(f"DB: `{client.path}`")
page = st.sidebar.radio("Navigate", ["Overview", "Factors", "Evaluations",
                                      "Lineage", "Deployments", "Events"])
st.sidebar.divider()
if st.sidebar.button("🔄 Refresh caches"):
    st.cache_data.clear()


# ============================================================
# Overview page
# ============================================================

if page == "Overview":
    st.title("Factor Hub — Overview")
    stats = client.stats()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Factors (active)",
              stats["factors"]["active"], delta=f"{stats['factors']['total']} total")
    c2.metric("Evaluations", stats["evaluations"]["total"],
              delta=f"ic: {stats['evaluations']['ic']}")
    c3.metric("Deployments (active)",
              stats["deployments"]["active"], delta=f"{stats['deployments']['total']} total")
    c4.metric("Events logged", stats["events"])

    st.divider()
    colA, colB = st.columns(2)

    with colA:
        st.subheader("Origin distribution")
        origins = pd.DataFrame(client.origin_distribution())
        if origins.empty:
            st.info("No factors yet. Run `python scripts/factor_lab.py hub migrate`.")
        else:
            st.bar_chart(origins.set_index("origin")["count"])
            st.dataframe(origins, hide_index=True, width="stretch")

    with colB:
        st.subheader("Top base features (dependents)")
        feats = pd.DataFrame(client.feature_deps(top_n=25))
        if feats.empty:
            st.info("Lineage not populated yet.")
        else:
            st.bar_chart(feats.set_index("feature")["dependents"])


# ============================================================
# Factors page
# ============================================================

elif page == "Factors":
    st.title("Factors")

    with st.container():
        f1, f2, f3, f4, f5 = st.columns([1.1, 1.1, 1.1, 1, 1])
        status = f1.selectbox("Status", ["active", "shadow", "deprecated",
                                          "candidate", "retired", ""], index=0)
        category = f2.text_input("Category", value="")
        origin = f3.text_input("Origin contains", value="")
        metric = f4.selectbox("Metric", ["oos_ic", "ic", "lgb_gain",
                                          "profit_pct", "sharpe"], index=0)
        ic_gt = f5.number_input(f"|{metric}| ≥", min_value=0.0, value=0.0, step=0.01)

    limit = st.slider("Max rows", 20, 500, 100, step=20)
    rows = client.query(
        status=status or None, category=category or None, origin=origin or None,
        metric_name=metric, ic_gt=ic_gt if ic_gt > 0 else None, limit=limit,
    )
    df = _attach_metric(client, rows, metric)
    st.caption(f"{len(df)} factors returned")
    st.dataframe(df, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Inspect / change status")
    if not df.empty:
        fid_choice = st.selectbox("Pick factor", df["id"].tolist(),
                                   format_func=lambda i: f"#{i} — {df.loc[df['id']==i,'name'].iloc[0]}")
        factor = client.get(int(fid_choice))
        if factor:
            st.code(factor.expression, language="python")
            st.json(asdict(factor))
            c1, c2 = st.columns(2)
            with c1:
                new_status = st.selectbox("New status", ["active", "shadow",
                                                           "deprecated", "candidate", "retired"])
                reason = st.text_input("Reason")
                if st.button("Apply status change"):
                    client.update_status(factor.id, new_status, reason=reason)
                    st.success(f"Set #{factor.id} → {new_status}")
            with c2:
                if st.button("🗑️ Delete factor", type="secondary"):
                    client.delete(factor.id)
                    st.warning(f"Deleted #{factor.id}")

    with st.expander("➕ Propose a new factor"):
        with st.form("propose"):
            name = st.text_input("Name (optional)")
            expression = st.text_area("Expression", height=80)
            category = st.text_input("Category", "mean_reversion")
            origin = st.text_input("Origin", "ui")
            description = st.text_area("Description", height=60)
            submitted = st.form_submit_button("Create")
            if submitted and expression.strip():
                fid = client.propose(
                    expression=expression, name=name or None,
                    category=category, origin=origin,
                    description=description, status="candidate",
                )
                st.success(f"Created factor #{fid}")


# ============================================================
# Evaluations page
# ============================================================

elif page == "Evaluations":
    st.title("Evaluations")

    rows = client.query(status="active", limit=200)
    if not rows:
        st.info("No factors yet.")
    else:
        df_f = pd.DataFrame(rows)
        fid = st.selectbox("Factor", df_f["id"].tolist(),
                            format_func=lambda i: f"#{i} — {df_f.loc[df_f['id']==i,'name'].iloc[0]}")
        metric = st.selectbox("Metric", ["oos_ic", "ic", "lgb_gain", "profit_pct", "sharpe"])
        evals = client.evaluations(int(fid), metric_name=metric, limit=500)
        if not evals:
            st.info(f"No {metric} evaluations for factor #{fid}.")
        else:
            edf = pd.DataFrame([asdict(e) for e in evals])
            edf["eval_at"] = pd.to_datetime(edf["eval_at"])
            edf = edf.sort_values("eval_at")
            st.line_chart(edf.set_index("eval_at")["metric_value"])
            st.dataframe(edf[["id", "eval_at", "eval_type", "metric_name", "metric_value",
                              "n_samples", "sign_agree", "period_start", "period_end", "notes"]],
                          hide_index=True, width="stretch")


# ============================================================
# Lineage page
# ============================================================

elif page == "Lineage":
    st.title("Feature lineage")
    feats = pd.DataFrame(client.feature_deps(top_n=100))
    if feats.empty:
        st.info("No lineage — run `hub migrate` first.")
    else:
        st.dataframe(feats, hide_index=True, width="stretch")

        chosen = st.selectbox("Show factors referencing:", feats["feature"].tolist())
        consumers = client.factors_by_feature(chosen)
        st.caption(f"{len(consumers)} active factors use `{chosen}`")
        if consumers:
            cdf = pd.DataFrame([asdict(f) for f in consumers])[
                ["id", "name", "category", "origin", "expression"]
            ]
            st.dataframe(cdf, hide_index=True, width="stretch")


# ============================================================
# Deployments page
# ============================================================

elif page == "Deployments":
    st.title("Deployments")

    active = client.active_deployment("production")
    if active:
        st.success(f"Active production deployment #{active.id} — "
                   f"{len(active.factor_ids)} factors, at {active.deployed_at}")
    else:
        st.warning("No active production deployment.")

    dps = client.deployments(limit=50)
    if dps:
        ddf = pd.DataFrame([asdict(d) for d in dps])
        st.dataframe(
            ddf[["id", "name", "deployed_at", "is_active", "factor_ids", "deployed_by", "notes"]],
            hide_index=True, width="stretch",
        )

        chosen = st.selectbox("Activate deployment", ddf["id"].tolist())
        if st.button("Activate"):
            client.activate(int(chosen))
            st.success(f"Activated deployment #{chosen}")

    st.divider()
    st.subheader("Create a new deployment")
    with st.form("create_deployment"):
        name = st.text_input("Name", "production")
        raw_ids = st.text_input("Factor IDs (comma-separated)")
        activate = st.checkbox("Activate now", value=True)
        notes = st.text_input("Notes")
        submit = st.form_submit_button("Create")
        if submit and raw_ids.strip():
            fids = [int(x) for x in raw_ids.replace(" ", "").split(",") if x]
            did = client.deploy(name, fids, activate=activate, notes=notes)
            st.success(f"Created deployment #{did} with {len(fids)} factors")


# ============================================================
# Events page
# ============================================================

elif page == "Events":
    st.title("Event log")
    et = st.selectbox("Filter", ["(all)", "factor.created", "factor.updated",
                                   "factor.evaluated", "factor.status_changed",
                                   "deployment.switched",
                                   "mining.started", "mining.loop_completed", "mining.finished",
                                   "backtest.started", "backtest.finished"])
    limit = st.slider("Limit", 50, 2000, 300, step=50)
    evts = client.events(limit=limit, event_type=(None if et == "(all)" else et))
    if not evts:
        st.info("No events yet.")
    else:
        edf = pd.DataFrame([asdict(e) for e in evts])
        edf = edf.sort_values("id", ascending=False)
        st.dataframe(edf[["id", "event_type", "factor_id", "created_at", "payload"]],
                      hide_index=True, width="stretch")
