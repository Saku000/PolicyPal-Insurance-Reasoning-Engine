import os
import json
from typing import List

from core import semantic_search
from config import CHROMA_PERSIST_DIR
from openai import OpenAI

client = OpenAI()

COMPARE_TOP_K = 12


def search_policy(policy_name: str, question: str):

    index_dir = os.path.join(CHROMA_PERSIST_DIR, policy_name)

    results = semantic_search(
        question,
        top_k=COMPARE_TOP_K,
        index_dir=index_dir,
        use_prefilter=True,
        min_hits=1
    )

    evidence = []

    for r in results:
        text = r.get("text", "")
        doc = r.get("doc_id", "unknown")

        evidence.append(f"[{doc}] {text}")

    return "\n\n".join(evidence)


def build_prompt(question, evidence_a, evidence_b, policy_a, policy_b):

    return f"""
You are an insurance policy analyst.

Compare the following two insurance policies using the provided evidence.

If information is incomplete, infer reasonably from the policy wording.

Provide:

1) A comparison table
2) A short summary

Question:
{question}

Policy A: {policy_a}
Evidence:
{evidence_a}

Policy B: {policy_b}
Evidence:
{evidence_b}

Output format:

Comparison Table:

| Feature | {policy_a} | {policy_b} |
|---|---|---|
| Coverage Limits | | |
| Deductible | | |
| Exclusions | | |
| Claim Conditions | | |
| Premium | | |

Summary:
"""


def compare_two_policies(question, policy_a, policy_b):

    evidence_a = search_policy(policy_a, question)
    evidence_b = search_policy(policy_b, question)

    prompt = build_prompt(
        question,
        evidence_a,
        evidence_b,
        policy_a,
        policy_b
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert insurance analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content