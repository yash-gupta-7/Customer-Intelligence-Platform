"""
rag/generator.py — Answer generation with evidence citation.

Supports two modes:
  - "local" mode: extractive summarisation (no LLM needed, always works)
  - "openai" mode: GPT-based generation with cited record IDs in the prompt
"""
import re
from collections import Counter
from loguru import logger

from app.config import get_settings

settings = get_settings()


# ── Theme Extraction ──────────────────────────────────────────────────────────

COMPLAINT_DOMAIN_TERMS = {
    "complaint", "complaints", "billing", "account", "credit", "loan", "mortgage",
    "fraud", "dispute", "refund", "charge", "fee", "bank", "card", "payment",
    "customer", "service", "unauthorized", "debt", "collection",
}


def _meaningful_words(text: str) -> set[str]:
    return {w for w in re.sub(r"[^a-z ]", "", text.lower()).split() if len(w) > 3}


def _question_domain_hits(question: str) -> int:
    q = question.lower()
    return sum(1 for term in COMPLAINT_DOMAIN_TERMS if term in q)


def _max_lexical_overlap(question: str, chunks: list[dict]) -> float:
    q_words = _meaningful_words(question)
    if not q_words or not chunks:
        return 0.0
    overlaps = []
    for chunk in chunks:
        text_words = _meaningful_words(chunk.get("text", ""))
        overlaps.append(len(q_words & text_words) / len(q_words))
    return max(overlaps)


THEME_KEYWORDS = {
    "Billing Issues": ["charge", "fee", "bill", "overcharge", "refund", "payment"],
    "Account Access": ["login", "locked", "password", "access", "account", "blocked"],
    "Customer Service": ["rude", "unhelpful", "wait", "hold", "agent", "representative"],
    "Fraud & Security": ["fraud", "scam", "unauthorized", "stolen", "identity", "hack"],
    "Loan & Credit": ["loan", "interest", "mortgage", "credit", "debt", "apr"],
    "Technical Issues": ["app", "website", "error", "crash", "technical", "system"],
    "Product Issues": ["product", "service", "feature", "policy", "terms", "condition"],
    "Dispute Resolution": ["dispute", "complaint", "escalate", "resolve", "unfair", "wrong"],
}


def extract_themes(texts: list[str]) -> list[str]:
    """Identify dominant complaint themes by keyword frequency."""
    combined = " ".join(texts).lower()
    theme_scores: dict[str, int] = {}
    for theme, keywords in THEME_KEYWORDS.items():
        score = sum(combined.count(kw) for kw in keywords)
        if score > 0:
            theme_scores[theme] = score

    sorted_themes = sorted(theme_scores.items(), key=lambda x: -x[1])
    return [t for t, _ in sorted_themes[:5]]


# ── Local Extractive Generator ────────────────────────────────────────────────

def _extractive_answer(question: str, chunks: list[dict]) -> tuple[str, float]:
    """
    Build a concise answer by selecting the most relevant sentences from
    retrieved chunks and formatting them into a structured response.
    Returns (answer_text, confidence_score).
    """
    if not chunks:
        return "No relevant complaint records found for the given query.", 0.0

    # Score each chunk by keyword overlap with question
    q_words = set(re.sub(r"[^a-z ]", "", question.lower()).split())
    scored = []
    for chunk in chunks:
        text = chunk.get("text", "")
        text_words = set(re.sub(r"[^a-z ]", "", text.lower()).split())
        overlap = len(q_words & text_words) / max(len(q_words), 1)
        scored.append((overlap + chunk.get("similarity_score", 0.0), text, chunk))

    scored.sort(key=lambda x: -x[0])
    top = scored[:3]

    # Build structured answer
    summary_parts = []
    for i, (_, text, chunk) in enumerate(top, 1):
        product = chunk.get("product", "N/A")
        issue = chunk.get("issue", "N/A")
        record_id = chunk.get("record_id", "N/A")
        snippet = text[:300].strip()
        if len(text) > 300:
            snippet += "..."
        summary_parts.append(
            f"[{i}] Record {record_id} ({product} / {issue}): {snippet}"
        )

    answer = (
        f"Based on {len(chunks)} retrieved complaint records:\n\n"
        + "\n\n".join(summary_parts)
        + "\n\nThe above records represent the most relevant complaints matching your query."
    )

    avg_sim = sum(c.get("similarity_score", 0.0) for c in chunks) / len(chunks)
    confidence = min(0.95, avg_sim * 1.2)

    return answer, round(confidence, 4)


# ── OpenAI Generator ─────────────────────────────────────────────────────────

def _openai_answer(question: str, chunks: list[dict]) -> tuple[str, float]:
    """GPT-based generation. Falls back to extractive if API key missing."""
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY not set — falling back to extractive mode.")
        return _extractive_answer(question, chunks)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        context = "\n\n".join(
            f"[Record {c['record_id']}] {c['text'][:400]}" for c in chunks[:5]
        )
        system_prompt = (
            "You are a complaint intelligence analyst. Answer the user's question "
            "using ONLY the provided complaint records. Always cite record IDs in "
            "square brackets like [REC-123]. Be concise and factual."
        )
        user_prompt = f"Question: {question}\n\nComplaint Records:\n{context}"

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        answer = response.choices[0].message.content.strip()
        return answer, 0.90

    except Exception as e:
        logger.warning(f"OpenAI call failed ({e}) — falling back to extractive mode.")
        return _extractive_answer(question, chunks)


# ── Public interface ──────────────────────────────────────────────────────────

def generate_answer(question: str, retrieved_chunks: list[dict]) -> tuple[str, list[str], float]:
    """
    Generate answer + extract themes + return cited record IDs.
    Returns: (answer, complaint_themes, confidence_score)
    """
    refusal_msg = (
        "I am sorry, but I could not find any relevant, high-confidence complaint records "
        "matching your query. I must decline to answer questions outside our complaint database."
    )

    if not retrieved_chunks:
        return refusal_msg, [], 0.0

    max_sim = max(c.get("similarity_score", 0.0) for c in retrieved_chunks)
    lexical_overlap = _max_lexical_overlap(question, retrieved_chunks)
    domain_hits = _question_domain_hits(question)

    # Reject off-topic questions that only weakly match complaint embeddings.
    if domain_hits == 0 and lexical_overlap < 0.12 and max_sim < 0.55:
        return refusal_msg, [], 0.0
    if max_sim < 0.35:
        return refusal_msg, [], 0.0

    texts = [c.get("text", "") for c in retrieved_chunks]
    themes = extract_themes(texts)

    if settings.llm_provider == "openai":
        answer, confidence = _openai_answer(question, retrieved_chunks)
    else:
        answer, confidence = _extractive_answer(question, retrieved_chunks)

    return answer, themes, confidence
