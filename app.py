
"""
Streamlit UI for the LangGraph claims-processing workflow (claims_agent.py).
 
Run:
    pip install -r requirements.txt
    streamlit run app.py
 
Keep app.py, claims_agent.py, membership_handbook.pdf and dummy.csv in the same
folder (or upload the handbook / customer CSV from the sidebar).
"""
import hashlib
import json
import os
import tempfile
import traceback
from pathlib import Path
 
import pandas as pd
import streamlit as st
 
import claims_agent as ca
 
# Guard: app.py and claims_agent.py must be the matching versions.
_REQUIRED_API = ["DATE_COLUMNS", "load_customers", "parse_date_all", "names_match", "ClaimsWorkflow"]
_missing_api = [n for n in _REQUIRED_API if not hasattr(ca, n)]
if _missing_api:
    st.set_page_config(page_title="Claims Processing Agent", page_icon="🛡️")
    st.error(f"`claims_agent.py` is out of date (missing: {', '.join(_missing_api)}). "
             f"The file being imported is:\n\n`{getattr(ca, '__file__', '?')}`")
    st.info("Replace it with the latest claims_agent.py, then stop Streamlit (Ctrl+C) and start it "
            "again with `streamlit run app.py` so Python does not reuse the old module.")
    st.stop()
 
APP_DIR = Path(__file__).resolve().parent
ca.INDEX_DIR = APP_DIR / "kb_index"           # local FAISS index
ca.LOG_PATH = APP_DIR / "claims_log.json"     # duplicate-check ledger
UPLOAD_DIR = Path(tempfile.gettempdir()) / "claims_app_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
 
DEFAULT_HANDBOOK = APP_DIR / "membership_handbook.pdf"
DEFAULT_CUSTOMERS = APP_DIR / "dummy.csv"
SAMPLE_BILLS_DIR = APP_DIR / "bills"
 
st.set_page_config(page_title="Claims Processing Agent", page_icon="🛡️", layout="wide")
 
NODE_LABELS = {
    "ingest_bill": "📄 Read bill",
    "verify_customer": "🔎 Verify customer",
    "reject_unknown_customer": "⛔ Rejected (no LLM)",
    "extract_details": "🤖 Extract line items (LLM)",
    "apply_rules": "📏 Apply business rules",
    "decide_by_rules": "⚙️ Decided by rules engine",
    "retrieve_policy": "📚 Retrieve handbook passages",
    "adjudicate": "🧑‍⚖️ Adjudicate (LLM)",
    "finalize": "✅ Build report",
}
 
 
# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save_upload(data: bytes, name: str) -> Path:
    """Persist an uploaded file to a temp path (the workflow works with file paths)."""
    digest = hashlib.sha256(data).hexdigest()[:10]
    path = UPLOAD_DIR / f"{digest}_{Path(name).name}"
    if not path.exists():
        path.write_bytes(data)
    return path
 
 
@st.cache_resource(show_spinner="Loading workflow...")
def get_workflow(handbook: str, customers: str, rebuild_token: int):
    """Cached so the embedding model / FAISS index are loaded once, not on every rerun."""
    wf = ca.ClaimsWorkflow(handbook, customers, persist=False, rebuild_index=rebuild_token > 0)
    return wf, wf.build()
 
 
def fmt_money(v) -> str:
    return "-" if v is None else f"${float(v):,.2f}"
 
 
def run_claim(graph, bill_path: Path, status) -> dict:
    """Stream the graph node by node so the UI can show live progress."""
    init = {"bill_path": str(bill_path), "llm_calls": 0, "trace": []}
    out = {"report": None, "bill_text": "", "identity": {}, "context": []}
    for chunk in graph.stream(init, stream_mode="updates"):
        for node, upd in chunk.items():
            upd = upd or {}
            detail = (upd.get("trace") or [""])[0]
            status.write(f"{NODE_LABELS.get(node, node)}  \n<small>{detail}</small>",
                         unsafe_allow_html=True)
            if node == "ingest_bill":
                out["bill_text"] = upd.get("bill_text", "")
                out["identity"] = upd.get("identity", {})
            elif node == "retrieve_policy":
                out["context"] = upd.get("retrieved_context", [])
            elif node == "finalize":
                out["report"] = upd.get("final_report")
    return out
 
 
def render_result(i: int, res: dict) -> None:
    if res.get("error"):
        st.error(f"**{res['error_type']}**: {res['error']}")
        if "onnection" in res["error_type"] or "onnection" in res["error"]:
            st.info("This is a network/SSL problem reaching the LLM API (proxy, VPN, firewall), "
                    "not a problem with the bill. Bills for customers that are not in the CSV are "
                    "rejected without any LLM call, so you can still test that path.")
        with st.expander("Traceback"):
            st.code(res["traceback"])
        return
 
    r = res["report"]
    verdict, status = r["verdict"], r["claim_status"]
    banner = f"### {verdict}  ·  status: `{status}`"
    if verdict == "ACCEPTED":
        st.success(banner)
    elif status == "Under_Review":
        st.warning(banner)
    else:
        st.error(banner)
    st.write(r["summary"])
 
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Claim ID", r["claim_id"] or "n/a")
    m2.metric("Amount claimed", fmt_money(r["amount_claimed"]))
    m3.metric("Approved amount", fmt_money(r["approved_amount"]))
    m4.metric("LLM calls", r["llm_calls"])
    st.caption(f"Decided by: {r['decided_by']}")
 
    left, right = st.columns(2)
    with left:
        st.markdown("#### Reasons")
        for x in r["reasons"]:
            st.markdown(f"- {x}")
        if r["handbook_citations"]:
            st.markdown("**Handbook citations**")
            for x in r["handbook_citations"]:
                st.markdown(f"- {x}")
        if r["bill_warnings"]:
            st.markdown("**Bill warnings**")
            for w in r["bill_warnings"]:
                st.warning(w)
        if r["next_steps"]:
            st.markdown("**Next steps**")
            for s in r["next_steps"]:
                st.markdown(f"- {s}")
 
    with right:
        st.markdown("#### Customer")
        cust = r["customer"]
        check_keys = {"name_on_bill", "name_on_file", "name_match",
                      "dob_on_bill", "dob_on_file", "dob_match"}
        for k, v in cust.items():
            if k not in check_keys:
                st.markdown(f"- **{k}**: {v}")
        if "name_on_bill" in cust:
            st.markdown("**Identity check (bill vs customer file)**")
            st.dataframe(pd.DataFrame([
                {"Field": "Name", "On bill": cust["name_on_bill"], "On file": cust["name_on_file"],
                 "Match": "✅" if cust["name_match"] else "❌"},
                {"Field": "Date of birth", "On bill": cust["dob_on_bill"], "On file": cust["dob_on_file"],
                 "Match": "✅" if cust["dob_match"] else "❌"},
            ]), hide_index=True)
        bill = r["bill"]
        if bill.get("line_items"):
            st.markdown("#### Bill details")
            st.markdown(f"- **Provider**: {bill.get('provider')}\n"
                        f"- **Invoice**: {bill.get('invoice_no')}\n"
                        f"- **Service date**: {bill.get('service_date')}\n"
                        f"- **Reason**: {bill.get('reason_for_claim')}")
            df = pd.DataFrame(bill["line_items"]).rename(columns={
                "cpt_code": "CPT", "description": "Description", "quantity": "Qty",
                "billed_amount": "Billed", "insurance_adjustment": "Ins. adj.",
                "patient_responsibility": "Patient resp."})
            st.dataframe(df, hide_index=True)
            t = bill.get("totals") or {}
            st.caption(f"Total billed {fmt_money(t.get('total_billed'))} · adjustments "
                       f"{fmt_money(t.get('insurance_adjustments'))} · insurance paid "
                       f"{fmt_money(t.get('insurance_paid'))} · balance due "
                       f"{fmt_money(t.get('patient_balance_due'))}")
        else:
            st.markdown("#### Identity read from bill")
            for k, v in (res.get("identity") or {}).items():
                st.markdown(f"- **{k}**: {v}")
 
    if res.get("context"):
        with st.expander("Handbook passages retrieved from the local index"):
            for n, c in enumerate(res["context"], 1):
                st.markdown(f"**[{n}] {c['source']} · p.{c['page']}** (distance {c['score']:.3f})")
                st.text(c["text"])
    with st.expander("Text extracted from the bill PDF"):
        st.text(res.get("bill_text") or "")
    with st.expander("Processing trace"):
        for t in r["trace"]:
            st.markdown(f"- {t}")
    with st.expander("Full report (JSON)"):
        st.json(r)
    st.download_button("Download report (JSON)", json.dumps(r, indent=2, default=str),
                       file_name=f"claim_report_{i + 1}.json", mime="application/json",
                       key=f"dl_{i}")
 
 
# --------------------------------------------------------------------------- #
# Sidebar - configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
 
    st.subheader("LLM")
    provider = st.selectbox("Provider", ["anthropic", "openai"],
                            index=0 if ca.LLM_PROVIDER != "openai" else 1)
    model = st.text_input("Model (blank = default)", value=ca.LLM_MODEL or "",
                          placeholder="claude-sonnet-5-5" if provider == "anthropic" else "gpt-4o-mini")
    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    key_in = st.text_input(f"{env_var}", type="password",
                           help="Only held in this server process's environment; never written to disk.")
    if key_in:
        os.environ[env_var] = key_in
    have_key = bool(os.environ.get(env_var))
    st.caption("✅ API key available" if have_key else
               "⚠️ No API key set - only the 'customer not found' path will work.")
 
    st.subheader("Knowledge base")
    hb_up = st.file_uploader("Handbook PDF (optional)", type="pdf", key="hb")
    if hb_up:
        handbook_path = save_upload(hb_up.getvalue(), hb_up.name)
        st.caption(f"Using uploaded: {hb_up.name}")
    else:
        handbook_path = DEFAULT_HANDBOOK
        st.caption(f"Using default: {DEFAULT_HANDBOOK.name}")
    if "rebuild_token" not in st.session_state:
        st.session_state.rebuild_token = 0
    if st.button("Rebuild embeddings index"):
        st.session_state.rebuild_token += 1
        get_workflow.clear()
        st.toast("Index will be rebuilt on the next run")
 
    st.subheader("Customers")
    cu_up = st.file_uploader("Customer CSV (optional)", type="csv", key="cu")
    if cu_up:
        customers_path = save_upload(cu_up.getvalue(), cu_up.name)
        st.caption(f"Using uploaded: {cu_up.name}")
    else:
        customers_path = DEFAULT_CUSTOMERS
        st.caption(f"Using default: {DEFAULT_CUSTOMERS.name}")
    if Path(customers_path).exists():
        try:
            cust_rows = ca.load_customers(str(customers_path))
        except Exception as e:
            st.error(f"Customer CSV problem: {e}")
        else:
            bad = [f"{r.get('cust_id')}: {col} = {r.get(col + '_raw')!r}"
                   for r in cust_rows for col in ca.DATE_COLUMNS
                   if not ca.parse_date_all(r.get(col + "_raw"))]
            if bad:
                st.warning("Unreadable dates in the customer CSV:\n\n" + "\n".join(f"- {b}" for b in bad))
            with st.expander("Preview customers (as read by the agent)"):
                df_c = pd.DataFrame(cust_rows)
                st.dataframe(df_c[[c for c in df_c.columns if not c.endswith("_raw")]], hide_index=True)
 
    st.subheader("Claim ledger")
    persist = st.checkbox("Save decisions to ledger", value=False,
                          help="When on, decisions are stored and the same claim submitted again "
                               "within 30 days is flagged as a duplicate (Rule 1).")
    if ca.LOG_PATH.exists():
        with st.expander("View ledger"):
            st.json(json.loads(ca.LOG_PATH.read_text() or "[]"))
        if st.button("Clear ledger"):
            ca.LOG_PATH.unlink()
            st.rerun()
 
 
# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("🛡️ Insurance Claims Processing Agent")
st.caption("Upload a bill → the agent reads it, checks the customer file, applies the handbook "
           "rules and gives an ACCEPTED / REJECTED verdict.")
 
for label, p in (("Handbook", handbook_path), ("Customer CSV", customers_path)):
    if not Path(p).exists():
        st.error(f"{label} not found at `{p}`. Put it next to app.py or upload it in the sidebar.")
        st.stop()
 
bill_files = st.file_uploader("Upload bill(s) (PDF)", type="pdf", accept_multiple_files=True)
 
samples = []
if SAMPLE_BILLS_DIR.exists():
    names = sorted(p.name for p in SAMPLE_BILLS_DIR.glob("*.pdf"))
    if names:
        samples = st.multiselect("...or pick sample bills from the bills/ folder", names)
 
run_clicked = st.button("Process claim(s)", type="primary", disabled=not (bill_files or samples))
 
if run_clicked:
    jobs = [(f.name, save_upload(f.getvalue(), f.name)) for f in bill_files]
    jobs += [(n, SAMPLE_BILLS_DIR / n) for n in samples]
 
    wf, graph = get_workflow(str(handbook_path), str(customers_path),
                             st.session_state.rebuild_token)
    # Re-read settings on every run (cheap) without reloading the embedding model.
    wf.customers_csv = str(customers_path)
    wf.persist = persist
    wf._llm = None
    ca.LLM_PROVIDER = provider
    ca.LLM_MODEL = model.strip() or None
 
    results = []
    for name, path in jobs:
        with st.status(f"Processing {name} ...", expanded=True) as status:
            try:
                res = run_claim(graph, path, status)
                status.update(label=f"{name}: {res['report']['verdict']}",
                              state="complete", expanded=False)
            except Exception as e:  # show the error instead of crashing the app
                res = {"error": str(e), "error_type": type(e).__name__,
                       "traceback": traceback.format_exc()}
                status.update(label=f"{name}: failed", state="error")
        res["name"] = name
        results.append(res)
    st.session_state.results = results
 
results = st.session_state.get("results")
if results:
    st.divider()
    if len(results) > 1:
        st.subheader("Summary")
        rows = []
        for res in results:
            r = res.get("report")
            rows.append({
                "Bill": res["name"],
                "Customer": (r["customer"].get("name") or r["customer"].get("member_id")) if r else "-",
                "Verdict": r["verdict"] if r else "ERROR",
                "Status": r["claim_status"] if r else "-",
                "Claimed": r["amount_claimed"] if r else None,
                "Approved": r["approved_amount"] if r else None,
                "LLM calls": r["llm_calls"] if r else None,
                "Decided by": r["decided_by"] if r else res.get("error_type"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)
 
    tabs = st.tabs([res["name"] for res in results])
    for i, (tab, res) in enumerate(zip(tabs, results)):
        with tab:
            render_result(i, res)