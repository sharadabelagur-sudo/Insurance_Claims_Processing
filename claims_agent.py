#!/usr/bin/env python3
"""
Insurance Claims Processing - LangGraph multi-agent workflow
============================================================

Pipeline (each box is a LangGraph node):

  START
    |
  ingest_bill ........ regex/pypdf only: pulls identity + totals from the bill (NO LLM)
    |
  verify_customer .... looks the member up in dummy.csv (NO LLM)
    |--- not found / identity mismatch ---> reject_unknown_customer (NO LLM) --> finalize
    |
  extract_details .... [LLM agent 1] line items, amounts, reason for claim
    |
  apply_rules ........ deterministic handbook rules (duplicate, fast-pass, tiers, SLA,
    |                  policy status/coverage window, receipts, arithmetic checks)
    |--- hard denial / duplicate / fast-pass ---> decide_by_rules (NO LLM) --> finalize
    |
  retrieve_policy .... semantic search over the local FAISS index of the handbook
    |
  adjudicate ......... [LLM agent 2] acts as the insurance agent -> ACCEPTED / REJECTED
    |
  finalize ........... builds report, writes claim ledger, (stub) claimant notification
    |
  END

Usage:
  python claims_agent.py --bill bill.pdf --handbook membership_handbook.pdf --customers dummy.csv
  python claims_agent.py --draw          # print the graph as a Mermaid diagram

Env vars:
  ANTHROPIC_API_KEY  (default provider)        or  CLAIMS_LLM_PROVIDER=openai + OPENAI_API_KEY
  CLAIMS_LLM_MODEL   override the chat model name
  CLAIMS_EMBED_MODEL local sentence-transformers model (default all-MiniLM-L6-v2)
"""
import argparse
import csv
import hashlib
import io
import json
import operator
import os
import re
import sys
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pypdf import PdfReader

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
INDEX_DIR = Path("kb_index")            # local FAISS index lives here
LOG_PATH = Path("claims_log.json")      # ledger used by the duplicate-prevention rule
EMBED_MODEL = os.getenv("CLAIMS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.getenv("CLAIMS_LLM_PROVIDER", "anthropic").lower()
LLM_MODEL = os.getenv("CLAIMS_LLM_MODEL")


def get_llm():
    """Chat model used by the two LLM agents (extractor + adjudicator)."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=LLM_MODEL or "gpt-4o-mini", temperature=0)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=LLM_MODEL or "claude-sonnet-5-5", max_tokens=2000)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((p.extract_text() or "") for p in reader.pages)


_MONTH_FIRST = ["%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%m/%d/%y"]
_DAY_FIRST = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y"]
_OTHER_FORMATS = ["%d-%b-%Y", "%d %b %Y", "%b %d, %Y", "%B %d, %Y", "%d %B %Y"]


def parse_date_all(s, dayfirst: bool = False) -> list:
    """Every valid reading of a date string, most likely first.
    Handles ISO (1991-11-22), US (11/22/1991), day-first (22-11-1991), timestamps, etc."""
    if not s:
        return []
    s = re.split(r"[T ]\d{1,2}:\d{2}", str(s).strip())[0].strip()
    order = (_DAY_FIRST + _MONTH_FIRST) if dayfirst else (_MONTH_FIRST + _DAY_FIRST)
    found = []
    for fmt in ["%Y-%m-%d", "%Y/%m/%d"] + order + _OTHER_FORMATS:
        try:
            d = datetime.strptime(s, fmt).date()
        except ValueError:
            continue
        if d not in found:
            found.append(d)
    return found


def parse_date(s: Optional[str], dayfirst: bool = False) -> Optional[date]:
    found = parse_date_all(s, dayfirst)
    return found[0] if found else None


_TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof"}


def _clean_text(s) -> str:
    """NFKC-normalise and drop invisible control/format characters (zero-width spaces etc.)."""
    s = unicodedata.normalize("NFKC", str(s or ""))
    return "".join(ch for ch in s if ch.isspace() or not unicodedata.category(ch).startswith("C"))


def name_tokens(s) -> set:
    return {w for w in re.findall(r"[^\W_]+", _clean_text(s).casefold()) if w not in _TITLES}


def names_match(a, b) -> bool:
    """Case/spacing/punctuation-insensitive; 'Doe, Jane' == 'Jane Doe'; 'Jane A Doe' ~ 'Jane Doe'."""
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    small, big = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return len(small) >= 2 and small <= big


def normalize_id(s) -> str:
    return re.sub(r"[^0-9a-z]", "", _clean_text(s).casefold())


def parse_money(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    neg = "-" in s
    digits = re.sub(r"[^\d.]", "", s)
    if not digits:
        return None
    v = round(float(digits), 2)
    return -v if neg else v


def add_business_days(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


_LABELS = [
    "Patient Name", "Date of Birth", "Patient ID", "Service Date", "Payer Name",
    "Policy / Member ID", "Group Number", "Claim Reference", "Account / Invoice #",
    "Statement Date",
]


def _lbl(label: str) -> str:
    return re.escape(label).replace(r"\ ", r"\s+")


def extract_field(text: str, label: str) -> Optional[str]:
    """Grab the value after 'Label:' and stop at the next known label or end of line."""
    others = "|".join(_lbl(l) for l in _LABELS)
    pat = _lbl(label) + r"\s*:\s*(.+?)(?=\s+(?:" + others + r")\s*:|\s*$)"
    m = re.search(pat, text, re.I | re.M)
    return m.group(1).strip() if m else None


def money_after(text: str, label: str) -> Optional[float]:
    m = re.search(_lbl(label) + r"\s*:?\s*(-?\s*\$?\s*[\d,]+\.\d{2})", text, re.I)
    return parse_money(m.group(1)) if m else None


def tier_for(amount: float) -> str:
    if amount <= 500:
        return "Level 1 - Junior Adjuster"
    if amount <= 5000:
        return "Level 2 - Operations Manager"
    return "Level 3 - Executive Sign-off (+ Audit Log Entry)"


REQUIRED_CUSTOMER_COLUMNS = ["cust_id", "member_id", "full_name", "dob",
                             "policy_start_date", "policy_end_date", "policy_status"]
DATE_COLUMNS = ["dob", "policy_start_date", "policy_end_date"]


def _detect_dayfirst(values) -> bool:
    """A column is day-first if any value has a first number > 12 (e.g. 22-11-1991)."""
    for v in values:
        m = re.match(r"^\s*(\d{1,2})[/.\-](\d{1,2})[/.\-]\d{2,4}", v or "")
        if m and int(m.group(1)) > 12:
            return True
    return False


def load_customers(path: str) -> list[dict]:
    """Tolerant CSV loader: BOM/UTF-16/ANSI, comma/semicolon/tab delimiters, any header case or
    stray spaces, and Excel-reformatted dates (normalised to ISO; the original is kept in *_raw)."""
    raw = Path(path).read_bytes()
    text = None
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeError:
            continue
    if text is None:
        raise ValueError("Could not decode the customer CSV (try saving it as UTF-8).")
    first_line = text.splitlines()[0] if text.strip() else ""
    delim = max(",;\t|", key=first_line.count)
    rows = []
    for r in csv.DictReader(io.StringIO(text, newline=""), delimiter=delim):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in r.items() if k is not None}
        if any(row.values()):
            rows.append(row)
    if not rows:
        raise ValueError("The customer CSV has no data rows.")
    missing = [c for c in REQUIRED_CUSTOMER_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"Customer CSV is missing required column(s): {', '.join(missing)}. "
                         f"Found: {', '.join(rows[0].keys())}")
    for col in DATE_COLUMNS:
        dayfirst = _detect_dayfirst([r.get(col) for r in rows])
        for r in rows:
            r[col + "_raw"] = r.get(col, "")
            d = parse_date(r.get(col), dayfirst)
            if d:
                r[col] = d.isoformat()
    return rows


# --------------------------------------------------------------------------- #
# Structured-output schemas for the LLM agents
# --------------------------------------------------------------------------- #
class LineItem(BaseModel):
    cpt_code: str
    description: str
    quantity: int = 1
    billed_amount: float
    insurance_adjustment: float = Field(description="Negative or zero number as shown on the bill")
    patient_responsibility: float


class BillDetails(BaseModel):
    provider_name: str
    service_date: str = Field(description="MM/DD/YYYY")
    care_setting: str = Field(description="e.g. Emergency Department, Outpatient, Inpatient")
    line_items: list[LineItem]
    total_billed: float
    insurance_adjustments: float
    insurance_paid: float
    patient_balance_due: float
    reason_for_claim: str = Field(
        description="Plain-English summary of what care was provided and why, based ONLY on the "
                    "billed services. Do not invent diagnoses."
    )


class ClaimDecision(BaseModel):
    verdict: Literal["ACCEPTED", "REJECTED"]
    confidence: float = Field(ge=0, le=1)
    approved_amount: float = Field(description="amount_claimed if ACCEPTED, otherwise 0")
    summary: str = Field(description="2-3 sentence plain-English explanation for the claimant")
    reasons: list[str]
    handbook_citations: list[str] = Field(
        description="Handbook rules/sections relied on, e.g. 'Rule 3 (Tiered Escalation Matrix), p.2'"
    )


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #
class ClaimState(TypedDict, total=False):
    bill_path: str
    bill_text: str
    identity: dict
    totals: dict
    customer: Optional[dict]
    customer_status: str                    # FOUND | NOT_FOUND | IDENTITY_MISMATCH
    identity_check: dict                    # bill vs customer-file comparison
    bill_details: dict
    claim: dict
    rule_results: dict
    retrieved_context: list
    decision: dict
    final_report: dict
    llm_calls: int
    trace: Annotated[list, operator.add]    # nodes append to this


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #
class ClaimsWorkflow:
    def __init__(self, handbook_path: str, customers_csv: str,
                 persist: bool = True, rebuild_index: bool = False):
        self.handbook_path = handbook_path
        self.customers_csv = customers_csv
        self.persist = persist
        self.rebuild_index = rebuild_index
        self._store = None
        self._llm = None

    # ------------------------- infrastructure ---------------------------- #
    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def _vector_store(self):
        """Build (or load) the local FAISS index of the membership handbook."""
        if self._store is not None:
            return self._store
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        digest = hashlib.sha256(Path(self.handbook_path).read_bytes()).hexdigest()
        stamp = INDEX_DIR / "handbook.sha256"

        fresh = (INDEX_DIR.exists() and stamp.exists()
                 and stamp.read_text().strip() == digest and not self.rebuild_index)
        if fresh:
            self._store = FAISS.load_local(
                str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
            return self._store

        reader = PdfReader(self.handbook_path)
        pages = [
            Document(page_content=p.extract_text() or "",
                     metadata={"source": Path(self.handbook_path).name, "page": i + 1})
            for i, p in enumerate(reader.pages)
        ]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700, chunk_overlap=120, separators=["\n\n", "\n", ". ", " ", ""])
        chunks = splitter.split_documents(pages)
        self._store = FAISS.from_documents(chunks, embeddings)
        INDEX_DIR.mkdir(exist_ok=True)
        self._store.save_local(str(INDEX_DIR))
        stamp.write_text(digest)
        return self._store

    # --- claim ledger (for Rule 1: duplicate prevention) ---
    def _load_log(self) -> list[dict]:
        if LOG_PATH.exists():
            try:
                return json.loads(LOG_PATH.read_text())
            except json.JSONDecodeError:
                return []
        return []

    def _append_log(self, entry: dict) -> None:
        log = self._load_log()
        log.append(entry)
        LOG_PATH.write_text(json.dumps(log, indent=2))

    # ------------------------------ nodes -------------------------------- #
    def ingest_bill(self, state: ClaimState) -> dict:
        """Deterministic extraction of identity + totals. No LLM."""
        text = read_pdf_text(state["bill_path"])
        identity = {
            "patient_name": extract_field(text, "Patient Name"),
            "dob": extract_field(text, "Date of Birth"),
            "patient_id": extract_field(text, "Patient ID"),
            "service_date": extract_field(text, "Service Date"),
            "payer": extract_field(text, "Payer Name"),
            "member_id": extract_field(text, "Policy / Member ID"),
            "group_number": extract_field(text, "Group Number"),
            "claim_reference": extract_field(text, "Claim Reference"),
            "invoice_no": extract_field(text, "Account / Invoice #"),
        }
        totals = {
            "total_billed": money_after(text, "Total Billed Charges"),
            "insurance_adjustments": money_after(text, "Insurance Adjustments"),
            "insurance_paid": money_after(text, "Insurance Paid"),
            "patient_balance_due": money_after(text, "Patient Balance Due"),
        }
        return {"bill_text": text, "identity": identity, "totals": totals,
                "trace": [f"ingest_bill: member_id={identity['member_id']}, "
                          f"balance_due={totals['patient_balance_due']}"]}

    def verify_customer(self, state: ClaimState) -> dict:
        """Check the CSV customer master. No LLM."""
        ident = state["identity"]
        rows = load_customers(self.customers_csv)
        mid = normalize_id(ident.get("member_id"))
        bill_dobs = set(parse_date_all(ident.get("dob")))

        def file_dobs(r):
            return set(parse_date_all(r.get("dob_raw"))) | set(parse_date_all(r.get("dob")))

        match = None
        if mid:
            match = next((r for r in rows if normalize_id(r["member_id"]) == mid), None)
        if not match and ident.get("patient_name") and bill_dobs:
            match = next((r for r in rows if names_match(r["full_name"], ident["patient_name"])
                          and bill_dobs & file_dobs(r)), None)

        if not match:
            return {"customer": None, "customer_status": "NOT_FOUND",
                    "trace": ["verify_customer: NOT_FOUND"]}

        name_ok = names_match(match["full_name"], ident.get("patient_name"))
        dob_ok = bool(bill_dobs & file_dobs(match))
        check = {
            "name_on_bill": ident.get("patient_name"), "name_on_file": match["full_name"],
            "name_match": name_ok,
            "dob_on_bill": ident.get("dob"), "dob_on_file": match.get("dob_raw") or match.get("dob"),
            "dob_match": dob_ok,
        }
        if not (name_ok and dob_ok):
            return {"customer": None, "customer_status": "IDENTITY_MISMATCH", "identity_check": check,
                    "trace": [f"verify_customer: member id found but identity differs "
                              f"(name match={name_ok}: {check['name_on_bill']!r} vs {check['name_on_file']!r}; "
                              f"dob match={dob_ok}: {check['dob_on_bill']!r} vs {check['dob_on_file']!r})"]}

        return {"customer": match, "customer_status": "FOUND", "identity_check": check,
                "trace": [f"verify_customer: FOUND cust_id={match['cust_id']}"]}

    @staticmethod
    def route_after_customer(state: ClaimState) -> str:
        return "extract_details" if state["customer_status"] == "FOUND" else "reject_unknown_customer"

    def reject_unknown_customer(self, state: ClaimState) -> dict:
        """Direct rejection - the LLM is never called on this path."""
        status = state["customer_status"]
        if status == "NOT_FOUND":
            msg = "Customer not found in the customer master file."
        else:
            chk = state.get("identity_check") or {}
            problems = []
            if not chk.get("name_match"):
                problems.append(f"name on bill {chk.get('name_on_bill')!r} vs on file {chk.get('name_on_file')!r}")
            if not chk.get("dob_match"):
                problems.append(f"date of birth on bill {chk.get('dob_on_bill')!r} vs on file {chk.get('dob_on_file')!r}")
            msg = ("Member ID found but identity does not match our records ("
                   + "; ".join(problems) + ").")
        decision = {
            "verdict": "REJECTED", "claim_status": "Denied",
            "decided_by": "customer-verification (no LLM)",
            "summary": msg + " The claim cannot be processed.",
            "reasons": [msg], "handbook_citations": [], "approved_amount": 0.0,
        }
        return {"decision": decision, "claim": {},
                "trace": ["reject_unknown_customer: rejected without LLM"]}

    def extract_details(self, state: ClaimState) -> dict:
        """LLM agent 1 - structured extraction of line items, amounts and reason."""
        extractor = self.llm.with_structured_output(BillDetails)
        details: BillDetails = extractor.invoke([
            SystemMessage(content=(
                "You extract structured data from hospital itemized bills. Copy numbers exactly "
                "as printed. Keep the sign of adjustments/insurance payments as shown. Descriptions "
                "may wrap across lines - join them. Never invent a diagnosis.")),
            HumanMessage(content=f"Extract the data from this bill:\n\n{state['bill_text']}"),
        ])
        return {"bill_details": details.model_dump(),
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": [f"extract_details: {len(details.line_items)} line items extracted (LLM)"]}

    def apply_rules(self, state: ClaimState) -> dict:
        """Deterministic implementation of the handbook's business-rules matrix."""
        cust, det = state["customer"], state["bill_details"]
        ident, totals = state["identity"], state["totals"]

        service = parse_date(ident.get("service_date")) or parse_date(det.get("service_date"))
        amount = totals.get("patient_balance_due")
        if amount is None:
            amount = det["patient_balance_due"]
        amount = round(float(amount), 2)

        today = date.today()
        seq = len(self._load_log()) + 1
        claim = {
            "claim_id": f"CLM-{today.year}{seq:04d}",
            "claimant_id": cust["cust_id"],
            "claim_type": "Health",
            "amount_claimed": amount,
            "incident_date": service.isoformat() if service else None,
            "receipts_attached": [state["bill_path"]],
            "submitted_on": today.isoformat(),
        }

        hard_denials, flags, warnings = [], [], []

        # ---- schema / eligibility checks ----
        if amount <= 0:
            hard_denials.append("amount_claimed must be greater than 0.00 (nothing owed by the patient).")
        if amount > 25 and len(claim["receipts_attached"]) < 1:
            hard_denials.append("Receipts are mandatory when amount_claimed exceeds $25.00.")
        if cust.get("policy_status", "").upper() != "ACTIVE":
            hard_denials.append(f"Policy status is {cust.get('policy_status')} - claimant is not in good standing.")
        start, end = parse_date(cust.get("policy_start_date")), parse_date(cust.get("policy_end_date"))
        if service and start and service < start:
            hard_denials.append(f"Service date {service} is before policy start {start}.")
        if service and end and service > end:
            hard_denials.append(f"Service date {service} is after policy end {end}.")
        covered = [c.strip().lower() for c in cust.get("covered_claim_types", "").split("|") if c.strip()]
        if covered and claim["claim_type"].lower() not in covered:
            hard_denials.append(f"Plan {cust.get('plan_type')} does not cover claim type {claim['claim_type']}.")

        # ---- Rule 1: duplicate prevention ----
        dup = None
        if service:
            for c in self._load_log():
                if (c["claimant_id"] == claim["claimant_id"]
                        and abs(c["amount_claimed"] - amount) < 0.005
                        and c.get("claim_status") != "Denied"):
                    prev = parse_date(c["incident_date"])
                    if prev and abs((service - prev).days) <= 30:
                        dup = c
                        break
        if dup:
            flags.append("FLAG_DUPLICATE")

        # ---- Rule 2: fast-pass ----
        policy_age = (service - start).days if (service and start) else None
        fast_pass = bool(
            amount <= 100.00 and policy_age is not None and policy_age > 90
            and len(claim["receipts_attached"]) >= 1
        )

        # ---- Rule 3: tiered escalation ----
        tier = tier_for(amount)

        # ---- Rule 4: SLA ----
        sla = {
            "intake_validation_target": "< 60 seconds",
            "manual_adjudication_due": add_business_days(today, 5).isoformat() + " (escalates on day 6)",
            "payout_trigger_due_if_approved": add_business_days(today, 3).isoformat() + " (3 business days after approval)",
        }

        # ---- bill integrity checks (informational, passed to the adjudicator) ----
        tb = totals.get("total_billed") if totals.get("total_billed") is not None else det["total_billed"]
        adj = totals.get("insurance_adjustments") if totals.get("insurance_adjustments") is not None else det["insurance_adjustments"]
        paid = totals.get("insurance_paid") if totals.get("insurance_paid") is not None else det["insurance_paid"]
        if abs(tb + adj + paid - amount) > 0.01:
            warnings.append(f"Totals do not reconcile: billed {tb} + adjustments {adj} + insurance paid {paid} != balance {amount}.")
        items = det["line_items"]
        sum_billed = round(sum(i["billed_amount"] for i in items), 2)
        if abs(sum_billed - tb) > 0.01:
            warnings.append(f"Sum of line-item billed amounts ({sum_billed}) differs from Total Billed ({tb}).")
        sum_pr = round(sum(i["patient_responsibility"] for i in items), 2)
        if abs(sum_pr - amount) > 0.01:
            note = (f"Sum of line-level patient responsibility ({sum_pr}) differs from Patient Balance Due ({amount}).")
            if abs(sum_pr + paid - amount) < 0.01:
                note += " The difference equals the insurance payment already applied, so amount_claimed uses the balance due."
            warnings.append(note)

        rule_results = {
            "hard_denials": hard_denials,
            "flags": flags,
            "duplicate_of": dup["claim_id"] if dup else None,
            "fast_pass_eligible": fast_pass,
            "policy_age_days_at_service": policy_age,
            "approval_tier": tier,
            "sla": sla,
            "warnings": warnings,
        }
        return {"claim": claim, "rule_results": rule_results,
                "trace": [f"apply_rules: denials={len(hard_denials)}, flags={flags}, "
                          f"fast_pass={fast_pass}, tier={tier}"]}

    @staticmethod
    def route_after_rules(state: ClaimState) -> str:
        r = state["rule_results"]
        if r["hard_denials"] or r["flags"] or r["fast_pass_eligible"]:
            return "decide_by_rules"
        return "retrieve_policy"

    def decide_by_rules(self, state: ClaimState) -> dict:
        """Handbook rules that fully decide the claim without needing the LLM."""
        r, claim = state["rule_results"], state["claim"]
        if r["hard_denials"]:
            decision = {
                "verdict": "REJECTED", "claim_status": "Denied",
                "decided_by": "rules-engine (no LLM)",
                "summary": "The claim failed mandatory validation checks.",
                "reasons": r["hard_denials"],
                "handbook_citations": ["Core Data Schema (claimant_id / amount_claimed / receipts_attached)"],
                "approved_amount": 0.0,
            }
        elif "FLAG_DUPLICATE" in r["flags"]:
            decision = {
                "verdict": "REJECTED", "claim_status": "Under_Review",
                "decided_by": "rules-engine (no LLM)",
                "summary": "Possible duplicate of an earlier claim; held for manual review.",
                "reasons": [f"FLAG_DUPLICATE: same claimant, amount and incident date within 30 days of "
                            f"{r['duplicate_of']}."],
                "handbook_citations": ["Rule 1 (Duplicate Prevention)"],
                "approved_amount": 0.0,
            }
        else:  # fast-pass
            decision = {
                "verdict": "ACCEPTED", "claim_status": "Auto-Approved",
                "decided_by": "rules-engine (no LLM)",
                "summary": "Claim qualifies for fast-pass auto-approval.",
                "reasons": [f"Amount ${claim['amount_claimed']:.2f} <= $100.00, policy age "
                            f"{r['policy_age_days_at_service']} days > 90, receipt attached."],
                "handbook_citations": ["Rule 2 (Fast-Pass Auto-Approval)"],
                "approved_amount": claim["amount_claimed"],
            }
        return {"decision": decision, "trace": [f"decide_by_rules: {decision['verdict']} (no LLM)"]}

    def retrieve_policy(self, state: ClaimState) -> dict:
        """Semantic search over the local FAISS index of the handbook."""
        claim, det = state["claim"], state["bill_details"]
        query = (f"{claim['claim_type']} claim amount ${claim['amount_claimed']} approval authority "
                 f"escalation tier auto-approval receipts policy age duplicate SLA payout. "
                 f"Care provided: {det['reason_for_claim']}")
        hits = self._vector_store().similarity_search_with_score(query, k=5)
        ctx = [{"source": d.metadata.get("source"), "page": d.metadata.get("page"),
                "score": float(s), "text": d.page_content} for d, s in hits]
        return {"retrieved_context": ctx,
                "trace": [f"retrieve_policy: {len(ctx)} handbook chunks retrieved"]}

    def adjudicate(self, state: ClaimState) -> dict:
        """LLM agent 2 - acts as the insurance agent and gives the final verdict."""
        ctx = "\n\n".join(
            f"[{i + 1}] ({c['source']} p.{c['page']})\n{c['text']}"
            for i, c in enumerate(state["retrieved_context"]))
        cust = state["customer"]
        facts = {
            "claim": state["claim"],
            "member": {k: cust.get(k) for k in ("cust_id", "plan_type", "policy_status",
                                            "policy_start_date", "policy_end_date")},
            "bill": state["bill_details"],
            "deterministic_rule_results": state["rule_results"],
        }
        system = (
            "You are a senior insurance claims adjudicator. Decide whether the claim is ACCEPTED or "
            "REJECTED.\n"
            "- Base the decision ONLY on the claim facts, the deterministic rule results and the "
            "handbook excerpts provided. Cite handbook rules (e.g. 'Rule 3') in handbook_citations.\n"
            "- Customer verification, policy status, duplicate check and receipt checks are already "
            "done and authoritative; do not redo them.\n"
            "- Review the bill itself for material problems: CPT code vs description mismatches, "
            "implausible quantities, unreconciled arithmetic. Minor presentation quirks that the "
            "rule results explain should not cause rejection.\n"
            "- If the handbook is silent on medical coverage or exclusions for a service, do NOT "
            "invent exclusions - say the handbook is silent and decide on the rules that do apply.\n"
            "- Mention which approval tier the claim is routed to and the payout SLA when accepting.\n"
            "- approved_amount = amount_claimed if ACCEPTED, else 0."
        )
        human = (f"CLAIM FACTS (JSON):\n{json.dumps(facts, indent=2, default=str)}\n\n"
                 f"HANDBOOK EXCERPTS:\n{ctx}")
        result: ClaimDecision = self.llm.with_structured_output(ClaimDecision).invoke(
            [SystemMessage(content=system), HumanMessage(content=human)])
        d = result.model_dump()
        d["claim_status"] = "Approved" if d["verdict"] == "ACCEPTED" else "Denied"
        d["decided_by"] = "LLM adjudicator + handbook RAG"
        return {"decision": d, "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": [f"adjudicate: {d['verdict']} (confidence {d['confidence']:.2f}) (LLM)"]}

    def finalize(self, state: ClaimState) -> dict:
        d, claim, rules = state["decision"], state.get("claim") or {}, state.get("rule_results") or {}
        cust = state.get("customer")
        ident = state["identity"]

        next_steps = []
        if d["claim_status"] in ("Approved", "Auto-Approved"):
            next_steps.append(f"Assign to: {rules.get('approval_tier')}")
            next_steps.append("Trigger payout API by: " + rules["sla"]["payout_trigger_due_if_approved"])
        elif d["claim_status"] == "Under_Review":
            next_steps.append("Manual adjudication due: " + rules["sla"]["manual_adjudication_due"])

        report = {
            "claim_id": claim.get("claim_id"),
            "verdict": d["verdict"],
            "claim_status": d["claim_status"],
            "decided_by": d["decided_by"],
            "customer": ({"cust_id": cust["cust_id"], "name": cust["full_name"], "plan": cust.get("plan_type")}
                         if cust else {"member_id": ident.get("member_id"), "status": state["customer_status"],
                                       **(state.get("identity_check") or {})}),
            "bill": {
                "provider": (state.get("bill_details") or {}).get("provider_name"),
                "invoice_no": ident.get("invoice_no"),
                "service_date": ident.get("service_date"),
                "reason_for_claim": (state.get("bill_details") or {}).get("reason_for_claim"),
                "line_items": (state.get("bill_details") or {}).get("line_items"),
                "totals": state.get("totals"),
            },
            "amount_claimed": claim.get("amount_claimed"),
            "approved_amount": d.get("approved_amount", 0.0),
            "summary": d["summary"],
            "reasons": d["reasons"],
            "handbook_citations": d.get("handbook_citations", []),
            "bill_warnings": rules.get("warnings", []),
            "next_steps": next_steps,
            "llm_calls": state.get("llm_calls", 0),
            "trace": state.get("trace", []),
        }

        if self.persist and claim:
            self._append_log({**{k: claim[k] for k in ("claim_id", "claimant_id", "amount_claimed",
                                                       "incident_date", "submitted_on")},
                              "claim_status": d["claim_status"], "verdict": d["verdict"]})

        # Stub for the handbook's "webhook notification" requirement.
        print(f"[webhook stub] notify claimant -> claim {report['claim_id']} is {d['claim_status']}")
        return {"final_report": report, "trace": ["finalize: report built"]}

    # ------------------------------ graph -------------------------------- #
    def build(self):
        g = StateGraph(ClaimState)
        g.add_node("ingest_bill", self.ingest_bill)
        g.add_node("verify_customer", self.verify_customer)
        g.add_node("reject_unknown_customer", self.reject_unknown_customer)
        g.add_node("extract_details", self.extract_details)
        g.add_node("apply_rules", self.apply_rules)
        g.add_node("decide_by_rules", self.decide_by_rules)
        g.add_node("retrieve_policy", self.retrieve_policy)
        g.add_node("adjudicate", self.adjudicate)
        g.add_node("finalize", self.finalize)

        g.add_edge(START, "ingest_bill")
        g.add_edge("ingest_bill", "verify_customer")
        g.add_conditional_edges("verify_customer", self.route_after_customer, {
            "extract_details": "extract_details",
            "reject_unknown_customer": "reject_unknown_customer"})
        g.add_edge("extract_details", "apply_rules")
        g.add_conditional_edges("apply_rules", self.route_after_rules, {
            "decide_by_rules": "decide_by_rules",
            "retrieve_policy": "retrieve_policy"})
        g.add_edge("retrieve_policy", "adjudicate")
        g.add_edge("adjudicate", "finalize")
        g.add_edge("reject_unknown_customer", "finalize")
        g.add_edge("decide_by_rules", "finalize")
        g.add_edge("finalize", END)
        return g.compile()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def print_report(r: dict) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n CLAIM DECISION: {r['verdict']}   (status: {r['claim_status']})\n{bar}")
    print(f" Claim ID       : {r['claim_id']}")
    print(f" Customer       : {r['customer']}")
    print(f" Decided by     : {r['decided_by']}   |   LLM calls: {r['llm_calls']}")
    if r["bill"].get("service_date"):
        print(f" Provider       : {r['bill'].get('provider')}")
        print(f" Service date   : {r['bill']['service_date']}")
        print(f" Reason         : {r['bill'].get('reason_for_claim')}")
    if r["amount_claimed"] is not None:
        print(f" Amount claimed : ${r['amount_claimed']:.2f}   Approved: ${r['approved_amount']:.2f}")
    for li in (r["bill"].get("line_items") or []):
        print(f"   - {li['cpt_code']:<6} {li['description'][:48]:<48} x{li['quantity']}  "
              f"billed ${li['billed_amount']:.2f}")
    print(f"\n Summary: {r['summary']}")
    print(" Reasons:")
    for x in r["reasons"]:
        print(f"   * {x}")
    if r["handbook_citations"]:
        print(" Handbook citations: " + "; ".join(r["handbook_citations"]))
    if r["bill_warnings"]:
        print(" Bill warnings:")
        for w in r["bill_warnings"]:
            print(f"   ! {w}")
    if r["next_steps"]:
        print(" Next steps:")
        for s in r["next_steps"]:
            print(f"   > {s}")
    print("\n Trace:")
    for t in r["trace"]:
        print(f"   - {t}")
    print(bar)


def main() -> None:
    ap = argparse.ArgumentParser(description="LangGraph insurance claims processor")
    ap.add_argument("--bill", default="bill.pdf")
    ap.add_argument("--handbook", default="membership_handbook.pdf")
    ap.add_argument("--customers", default="dummy.csv")
    ap.add_argument("--rebuild-index", action="store_true", help="force re-embedding the handbook")
    ap.add_argument("--no-persist", action="store_true", help="do not write to claims_log.json")
    ap.add_argument("--json", action="store_true", help="print the final report as JSON")
    ap.add_argument("--draw", action="store_true", help="print the graph as Mermaid and exit")
    args = ap.parse_args()

    wf = ClaimsWorkflow(args.handbook, args.customers,
                        persist=not args.no_persist, rebuild_index=args.rebuild_index)
    app = wf.build()

    if args.draw:
        print(app.get_graph().draw_mermaid())
        return

    for p in (args.bill, args.handbook, args.customers):
        if not Path(p).exists():
            sys.exit(f"File not found: {p}")

    result = app.invoke({"bill_path": args.bill, "llm_calls": 0, "trace": []})
    report = result["final_report"]
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
