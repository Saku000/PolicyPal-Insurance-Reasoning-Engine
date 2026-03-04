# prod_compare.py
# Production compare: cached structured summaries (per policy) + compare summaries.
# Upgrade: Detect placeholder values (e.g., $000, $XXX, TBD, "see schedule") and treat as Missing.

import json
import os
import re
from typing import Dict, Any, Optional, List

import core
from policy_paths import COMPARE_DIR
from prod_retriever import retrieve_evidence

# Fixed K (NOT shown in UI)
COMPARE_TOP_K = 12

# ---------- field routing / query templates ----------
FIELD_QUERIES: Dict[str, List[str]] = {
    "coverage_limits": [
        "Limits of Liability",
        "liability limits",
        "Bodily Injury Liability",
        "Property Damage Liability",
        "each person",
        "each accident",
        "per accident",
        "limit of liability",
        "Uninsured Motorist",
        "Underinsured Motorist",
        "UM/UIM",
    ],
    "deductibles": [
        "Deductible",
        "collision deductible",
        "comprehensive deductible",
        "deductible applies",
        "Collision",
        "Comprehensive",
        "Other Than Collision",
    ],
    "exclusions": [
        "Exclusions",
        "We do not provide coverage",
        "This coverage does not apply",
        "is not covered",
        "not cover",
        "not pay",
    ],
    "claim_conditions": [
        "Duties after an accident",
        "Duties after loss",
        "Notice",
        "promptly notify",
        "cooperate",
        "proof of loss",
        "assist and cooperate",
        "claim reporting",
        "report the accident",
    ],
    "premium": [
        "Premium",
        "Total premium",
        "policy premium",
        "fees",
        "payment",
        "POLICY PREMIUM",
        "Declarations",
    ],
}


# ---------- helpers ----------
def _safe_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        name = "policy"
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name)


def _summary_path(policy_name: str) -> str:
    safe = _safe_name(policy_name)
    return os.path.join(str(COMPARE_DIR), f"{safe}__summary.json")


def _ensure_field_obj(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"value": None, "status": "missing", "evidence": []}
    value = obj.get("value", None)
    status = (obj.get("status") or "missing").lower()
    evidence = obj.get("evidence") or []
    if status not in ("found", "inferred", "missing"):
        status = "missing"
    if not isinstance(evidence, list):
        evidence = []
    return {"value": value, "status": status, "evidence": evidence}


# ---------- placeholder detection (NEW) ----------
_PLACEHOLDER_PATTERNS = [
    r"^\$?\s*0{2,}\s*$",                 # "000" / "$000" / " 000 "
    r"^\$?\s*x{2,}\s*$",                 # "XXX" / "$XXX"
    r"\bTBD\b",                          # TBD
    r"to be determined",                 # To be determined
    r"not provided",                     # not provided
    r"not specified",                    # not specified
    r"see (the )?(declarations|schedule)",  # see declarations/schedule
    r"refer to (the )?(declarations|schedule|endorsement)",  # refer to ...
    r"shown on (the )?(declarations|schedule)",              # shown on ...
]


def _is_placeholder_value(value: Any) -> bool:
    """
    Detect template / placeholder / deferred references that should be treated as Missing.
    IMPORTANT: We do NOT treat 'N/A' or 'Not Applicable' as placeholder;
    those can be a valid "not applicable" value (not missing).
    """
    if value is None:
        return False
    txt = str(value).strip()
    if not txt:
        return False

    low = txt.lower()

    # N/A is often a real meaning "not applicable" rather than missing
    if low in {"n/a", "na", "not applicable"}:
        return False

    # exact "$0" can be real deductible; do NOT blanket-mark as placeholder
    # but "$000" is extremely likely placeholder.
    if low in {"$0", "0", "$0.00", "0.00"}:
        return False

    # match common placeholder patterns
    for pat in _PLACEHOLDER_PATTERNS:
        if re.search(pat, txt, flags=re.IGNORECASE):
            return True

    # also catch obvious template tokens like "___"
    if "___" in txt or "____" in txt:
        return True

    return False


def _normalize_placeholders_in_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process LLM summary:
    if a field value looks like placeholder, force it to missing.
    """
    if not isinstance(summary, dict):
        return {"policy_name": "", "fields": {}}

    fields = summary.get("fields", {})
    if not isinstance(fields, dict):
        fields = {}

    for k in ["coverage_limits", "deductibles", "exclusions", "claim_conditions", "premium"]:
        fo = _ensure_field_obj(fields.get(k, {}))
        val = fo.get("value", None)

        if _is_placeholder_value(val):
            fo["value"] = None
            fo["status"] = "missing"
            ev = fo.get("evidence") or []
            ev = ev if isinstance(ev, list) else []
            ev.insert(0, "Detected placeholder / template value; treated as Missing.")
            fo["evidence"] = ev[:5]
            fields[k] = fo
        else:
            fields[k] = fo

    return {"policy_name": summary.get("policy_name", ""), "fields": fields}


def _render_value(field_obj: Dict[str, Any]) -> str:
    """
    For comparison table cell. Always return a string.
    Also enforce placeholder->Missing at render time (extra safety).
    """
    status = (field_obj.get("status") or "missing").lower()
    value = field_obj.get("value", None)

    if status == "missing" or value is None or str(value).strip() == "":
        return "Missing"

    # extra guard: placeholder values should display Missing
    if _is_placeholder_value(value):
        return "Missing"

    return str(value).strip()


def _missing_fields(summary: Dict[str, Any]) -> List[str]:
    fields = (summary.get("fields") or {}) if isinstance(summary, dict) else {}
    missing = []
    for k in ["coverage_limits", "deductibles", "exclusions", "claim_conditions", "premium"]:
        fo = _ensure_field_obj(fields.get(k, {}))
        # placeholder guard
        if fo["status"] == "missing" or _is_placeholder_value(fo.get("value")):
            missing.append(k)
    return missing


# ---------- core: build structured summary (cached) ----------
def build_policy_summary(
    policy_name: str,
    store_path: str,
    api_key: Optional[str],
    force: bool = False,
) -> Dict[str, Any]:
    """
    Build (or load cached) structured summary for a policy.
    The summary is extracted ONLY from retrieved evidence.

    Production rules:
    - If value is not explicit -> missing
    - Never guess numeric values
    - Placeholder/template values -> missing
    """
    path = _summary_path(policy_name)
    if (not force) and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        return _normalize_placeholders_in_summary(loaded)

    # retrieve evidence per field (multi-query + hybrid search)
    evidence_by_field: Dict[str, List[Dict[str, Any]]] = {}
    for field, qs in FIELD_QUERIES.items():
        ev = retrieve_evidence(
            store_path=store_path,
            queries=qs,
            api_key=api_key,
            dense_top_k=12,
            bm25_top_k=8,
            final_k=COMPARE_TOP_K,
        )
        evidence_by_field[field] = ev

    # Compact evidence to control prompt size
    compact_evidence_by_field: Dict[str, List[Dict[str, Any]]] = {}
    for field, ev_list in evidence_by_field.items():
        compact_evidence_by_field[field] = []
        for ev in ev_list:
            meta = ev.get("metadata", {}) or {}
            compact_evidence_by_field[field].append(
                {
                    "doc_name": meta.get("doc_name", "unknown"),
                    "page_start": meta.get("page_start", -1),
                    "page_end": meta.get("page_end", -1),
                    "text": (ev.get("text", "") or "")[:1800],
                }
            )

    system = (
        "You are an expert insurance policy analyst.\n"
        "Extract a structured policy summary.\n\n"
        "STRICT RULES:\n"
        "1) Use ONLY the provided evidence.\n"
        "2) If a value is not explicitly stated, set value=null and status='missing'.\n"
        "3) Do NOT guess or infer ANY numeric values (limits, deductibles, premiums).\n"
        "4) IMPORTANT: If the value appears as a placeholder/template (e.g., '$000', '000', '$XXX', 'TBD', "
        "'see declarations', 'refer to schedule'), treat it as missing.\n"
        "5) You MAY set status='inferred' only for NON-numeric qualitative points (e.g., 'requires prompt notice'), "
        "and only if evidence clearly implies it.\n"
        "6) For each field, include up to 3 evidence snippets (doc+pages+short quote) in evidence[].\n"
        "7) Output MUST be valid JSON ONLY.\n"
    )

    user_obj = {
        "task": "build_policy_summary",
        "policy_name": policy_name,
        "fields": ["coverage_limits", "deductibles", "exclusions", "claim_conditions", "premium"],
        "evidence_by_field": compact_evidence_by_field,
        "output_schema": {
            "policy_name": "string",
            "fields": {
                "coverage_limits": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
                "deductibles": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
                "exclusions": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
                "claim_conditions": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
                "premium": {"value": "string|null", "status": "found|inferred|missing", "evidence": "string[]"},
            },
        },
        "formatting_guidance": {
            "coverage_limits": "Summarize major limits in 1-4 lines. If multiple coverages, separate with semicolons.",
            "deductibles": "List deductible amounts and which coverage they apply to. If not explicit or placeholder -> missing.",
            "exclusions": "List key exclusions as bullets in a single string OR semicolon-separated.",
            "claim_conditions": "Summarize key duties: prompt notice, cooperate, documentation, etc.",
            "premium": "If premium numbers are not explicit or placeholder -> missing. Do not guess.",
        },
    }

    client = core._openai_client(api_key)
    resp = client.chat.completions.create(
        model=core.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        temperature=0.1,
    )
    text = (resp.choices[0].message.content or "").strip()

    # Parse JSON with minimal salvage
    try:
        summary = json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            summary = json.loads(text[start : end + 1])
        else:
            summary = {"policy_name": policy_name, "fields": {}, "raw": text}

    # Normalize fields + placeholder post-processing
    if not isinstance(summary, dict):
        summary = {"policy_name": policy_name, "fields": {}}

    summary.setdefault("policy_name", policy_name)
    fields = summary.get("fields", {})
    if not isinstance(fields, dict):
        fields = {}

    norm_fields = {}
    for k in ["coverage_limits", "deductibles", "exclusions", "claim_conditions", "premium"]:
        norm_fields[k] = _ensure_field_obj(fields.get(k, {}))

    summary = {"policy_name": policy_name, "fields": norm_fields}
    summary = _normalize_placeholders_in_summary(summary)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ---------- compare summaries ----------
def compare_policies_prod(
    policy_a_name: str,
    policy_a_store: str,
    policy_b_name: str,
    policy_b_store: str,
    question: str,
    api_key: Optional[str],
    force_refresh_summaries: bool = False,
) -> str:
    """
    Production compare:
    1) Build/load cached structured summaries for A and B
    2) Compare summaries and enforce Missing values in table (including placeholders)
    """
    sum_a = build_policy_summary(policy_a_name, policy_a_store, api_key=api_key, force=force_refresh_summaries)
    sum_b = build_policy_summary(policy_b_name, policy_b_store, api_key=api_key, force=force_refresh_summaries)

    fa = sum_a.get("fields", {}) or {}
    fb = sum_b.get("fields", {}) or {}

    rows = [
        ("Coverage limits", _render_value(_ensure_field_obj(fa.get("coverage_limits"))), _render_value(_ensure_field_obj(fb.get("coverage_limits")))),
        ("Deductibles", _render_value(_ensure_field_obj(fa.get("deductibles"))), _render_value(_ensure_field_obj(fb.get("deductibles")))),
        ("Key exclusions", _render_value(_ensure_field_obj(fa.get("exclusions"))), _render_value(_ensure_field_obj(fb.get("exclusions")))),
        ("Claim conditions", _render_value(_ensure_field_obj(fa.get("claim_conditions"))), _render_value(_ensure_field_obj(fb.get("claim_conditions")))),
        ("Premium", _render_value(_ensure_field_obj(fa.get("premium"))), _render_value(_ensure_field_obj(fb.get("premium")))),
    ]

    table_md = f"| Feature | {policy_a_name} | {policy_b_name} |\n|---|---|---|\n"
    for feat, va, vb in rows:
        table_md += f"| {feat} | {va} | {vb} |\n"

    missing_a = _missing_fields(sum_a)
    missing_b = _missing_fields(sum_b)

    system = (
        "You are an expert insurance analyst.\n\n"
        "STRICT RULES:\n"
        "1) Use ONLY the provided summaries + the pre-built comparison table.\n"
        "2) Do NOT invent any missing values. If a value is Missing in the table, treat it as unknown.\n"
        "3) Do NOT claim 'No missing information' if any field is Missing.\n"
        "4) If a value appears as '$000', '000', '$XXX', 'TBD', or refers to 'see declarations' or 'see schedule', "
        "explain that this is likely a template placeholder or unfilled value rather than a real number.\n"
        "5) In the Missing Info Checklist, briefly explain why the information is missing when possible.\n"
        "6) Be concise and practical.\n"
    )

    user_obj = {
        "task": "compare_two_policies",
        "question": question,
        "policy_a_name": policy_a_name,
        "policy_b_name": policy_b_name,
        "comparison_table_markdown": table_md,
        "summaries": {"policy_a": sum_a, "policy_b": sum_b},
        "missing": {"policy_a": missing_a, "policy_b": missing_b},
        "required_sections": [
            "Direct Answer",
            "Key Differences",
            "Who Should Choose A vs B",
            "Missing Info Checklist",
        ],
        "output_requirements": {
            "must_include_table": True,
            "missing_keyword": "Missing",
        },
        "missing_explanation_rule":
            "If a field shows placeholder values such as '$000', explain that the value appears to be a template placeholder and the actual value is not specified."
    }

    client = core._openai_client(api_key)
    resp = client.chat.completions.create(
        model=core.CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_obj, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    text = (resp.choices[0].message.content or "").strip()

    # Ensure the output includes the table even if the model forgets
    if "| Feature |" not in text:
        text = "## Comparison Table\n\n" + table_md + "\n\n" + text

    return text