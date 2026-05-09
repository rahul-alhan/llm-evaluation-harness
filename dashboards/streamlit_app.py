"""Streamlit dashboard for browsing eval results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="LLM Eval Harness", layout="wide")
st.title("LLM Evaluation Harness")

results_dir = Path("results")
files = sorted(results_dir.glob("*.json"))

if not files:
    st.warning("No results found. Run `python -m harness.cli evaluate ...` first.")
    st.stop()

selection = st.sidebar.selectbox("Result file", files, format_func=lambda p: p.name)
data = json.loads(selection.read_text())

st.subheader(f"Prompt: `{data['prompt']}` (v{data.get('version', 'latest')})")
st.metric("Items evaluated", data["n_items"])
status = "PASS" if data["gates_passed"] else "FAIL"
st.metric("Gate status", status)

st.markdown("### Metrics")
metrics_df = pd.DataFrame(
    [{"metric": k, "score": float(v)} for k, v in data["metrics"].items()]
)
fig = px.bar(metrics_df, x="metric", y="score", text="score", range_y=[0, 1])
fig.update_traces(texttemplate="%{text:.3f}")
st.plotly_chart(fig, use_container_width=True)

if data["gate_failures"]:
    st.error("Gate failures:")
    for f in data["gate_failures"]:
        st.code(f)

st.markdown("### Sample drilldown")
for i, sample in enumerate(data["samples"]):
    with st.expander(f"Q{i+1}: {sample['question'][:80]}"):
        st.markdown("**Answer:**")
        st.write(sample["answer"])
        st.markdown("**Ground truth:**")
        st.write(sample["ground_truth"])
        if sample.get("contexts"):
            st.markdown("**Contexts:**")
            for c in sample["contexts"]:
                st.code(c[:500])
