"""
rag/eval_harness.py — RAG Evaluation Harness.
Runs 10 structured queries to check retrieval precision/overlap and tests refusal on out-of-domain questions.
Generates a markdown pass/fail summary report in reports/rag_eval_report.md.
"""
import os
import sys
import json
import pandas as pd
from loguru import logger

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import get_settings
from app.rag.pipeline import run_rag_query, build_rag_index
from app.rag.retriever import get_retriever

settings = get_settings()


def run_evaluation() -> dict:
    logger.info("Starting RAG Evaluation Harness...")

    # Ensure index exists/built
    retriever = get_retriever()
    if retriever.index is None or retriever.index.ntotal == 0:
        logger.info("RAG Index not found. Building FAISS index first...")
        build_rag_index()

    # Load complaints
    if not os.path.exists(settings.complaints_data_path):
        logger.error(f"Complaints file not found at {settings.complaints_data_path}")
        return {"status": "error", "message": "Complaints file missing."}

    df = pd.read_csv(settings.complaints_data_path)

    # Define 10 evaluation categories dynamically linked to actual complaint IDs in dataset
    eval_templates = [
        {"desc": "unexpected fee on credit card", "prod": "Credit card", "kw": "unexpected fee"},
        {"desc": "unexpected fee on checking account", "prod": "Checking account", "kw": "unexpected fee"},
        {"desc": "unexpected fee on student loan", "prod": "Student loan", "kw": "unexpected fee"},
        {"desc": "application denied on mortgage", "prod": "Mortgage", "kw": "denied without explanation"},
        {"desc": "application denied on student loan", "prod": "Student loan", "kw": "denied without explanation"},
        {"desc": "unauthorized charges on credit card", "prod": "Credit card", "kw": "unauthorized charges"},
        {"desc": "unauthorized charges on checking account", "prod": "Checking account", "kw": "unauthorized charges"},
        {"desc": "interest rate changed on mortgage", "prod": "Mortgage", "kw": "interest rate on my mortgage was changed"},
        {"desc": "interest rate changed on checking account", "prod": "Checking account", "kw": "interest rate on my checking account was changed"},
        {"desc": "resolve billing disputes", "prod": "Credit card", "kw": "billing disputes"},
    ]

    eval_cases = []
    for case in eval_templates:
        # Find a complaint matching product & narrative keyword
        match = df[
            (df["product"].str.lower() == case["prod"].lower()) &
            (df["consumer_complaint_narrative"].str.lower().str.contains(case["kw"].lower()))
        ]
        if not match.empty:
            row = match.iloc[0]
            eval_cases.append({
                "question": f"Detail the complaint for a {case['prod'].lower()} regarding {case['kw']}.",
                "product": case["prod"],
                "expected_id": str(row["complaint_id"]),
                "snippet": row["consumer_complaint_narrative"][:100],
                "type": "in-domain"
            })
        else:
            # Fallback if no matching synthetics, take the first matching product
            match_prod = df[df["product"].str.lower() == case["prod"].lower()]
            if not match_prod.empty:
                row = match_prod.iloc[0]
                eval_cases.append({
                    "question": f"Summarize the issues for product {case['prod']}.",
                    "product": case["prod"],
                    "expected_id": str(row["complaint_id"]),
                    "snippet": row["consumer_complaint_narrative"][:100],
                    "type": "in-domain"
                })

    # Add 2 out-of-domain refusal cases
    eval_cases.append({
        "question": "What is the capital of France?",
        "product": None,
        "expected_id": None,
        "snippet": "N/A",
        "type": "refusal"
    })
    eval_cases.append({
        "question": "Tell me a recipe for chocolate chip cookies.",
        "product": None,
        "expected_id": None,
        "snippet": "N/A",
        "type": "refusal"
    })

    results = []
    passed_count = 0

    for idx, case in enumerate(eval_cases, 1):
        q = case["question"]
        prod = case["product"]
        exp_id = case["expected_id"]
        case_type = case["type"]

        logger.info(f"Running query {idx}/{len(eval_cases)} [{case_type}]: '{q}'")

        try:
            response = run_rag_query(question=q, product=prod, top_k=5)
            retrieved_ids = response.cited_record_ids

            if case_type == "in-domain":
                # Check overlap / hit rate
                hit = exp_id in retrieved_ids
                overlap_score = 1.0 if hit else 0.0
                passed = hit
                status_str = "PASS" if passed else "FAIL"
                details = f"Expected ID: {exp_id}. Retrieved: {retrieved_ids}"
            else:
                # Refusal test: answer must indicate refusal, confidence must be 0.0
                refused = response.confidence_score == 0.0 or "decline to answer" in response.answer.lower() or "sorry" in response.answer.lower()
                passed = refused
                overlap_score = 0.0
                status_str = "PASS" if passed else "FAIL"
                details = f"Refusal verified: {passed} (Confidence={response.confidence_score})"

            if passed:
                passed_count += 1

            results.append({
                "id": idx,
                "question": q,
                "type": case_type,
                "expected_id": exp_id,
                "retrieved_ids": retrieved_ids,
                "overlap_score": overlap_score,
                "confidence": response.confidence_score,
                "status": status_str,
                "details": details
            })

        except Exception as e:
            logger.exception(f"Error evaluating query {idx}: {e}")
            results.append({
                "id": idx,
                "question": q,
                "type": case_type,
                "expected_id": exp_id,
                "retrieved_ids": [],
                "overlap_score": 0.0,
                "confidence": 0.0,
                "status": "FAIL",
                "details": f"Execution failed: {str(e)}"
            })

    total_tests = len(eval_cases)
    pass_rate = (passed_count / total_tests) * 100

    report_data = {
        "total_tests": total_tests,
        "passed_count": passed_count,
        "pass_rate": pass_rate,
        "results": results
    }

    # Generate Markdown Report
    report_markdown = f"""# RAG Evaluation Harness Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance Summary

- **Total Tests:** {total_tests}
- **Tests Passed:** {passed_count}
- **Pass Rate:** {pass_rate:.1f}%
- **Overall Status:** {"PASS" if pass_rate >= 80.0 else "FAIL"}

---

## Detailed Results

| ID | Query | Type | Expected ID | Retrieved IDs | Confidence | Status | Details |
|---|---|---|---|---|---|---|---|
"""

    for r in results:
        ret_str = ", ".join(r["retrieved_ids"]) if r["retrieved_ids"] else "None"
        report_markdown += f"| {r['id']} | {r['question']} | {r['type']} | {r['expected_id'] or 'N/A'} | {ret_str} | {r['confidence']:.2f} | **{r['status']}** | {r['details']} |\n"

    # Save markdown report
    os.makedirs("reports", exist_ok=True)
    with open("reports/rag_eval_report.md", "w") as f:
        f.write(report_markdown)

    logger.info(f"Evaluation complete. Pass rate: {pass_rate:.1f}%. Report saved to reports/rag_eval_report.md")
    return report_data


if __name__ == "__main__":
    run_evaluation()
