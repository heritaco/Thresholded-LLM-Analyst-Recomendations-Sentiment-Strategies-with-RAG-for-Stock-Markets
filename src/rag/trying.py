import json
from typing import Any, Dict, Optional, List

from pathlib import Path
import pandas as pd

try:
    import ollama  # pip install ollama
except ImportError:
    ollama = None
    print("Warning: 'ollama' is not installed. Run `pip install ollama` to use the sentiment functions.")

import numpy as np
from dataclasses import dataclass

from langchain_ollama import OllamaEmbeddings 



import os
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")


SENTIMENT_SYSTEM_PROMPT = """\
You are a financial news sentiment analyst focusing on a single stock.

Given a short headline or short blurb about Apple Inc. (ticker: AAPL),
classify the sentiment with respect to Apple's FUTURE stock performance as:

- "bullish": mainly positive impact on AAPL
- "bearish": mainly negative impact on AAPL
- "neutral": mixed or unclear impact, or Apple only mentioned tangentially

Return a compact JSON object with keys:
- "sentiment": one of ["bullish", "bearish", "neutral"]
- "score": a number between -1 and 1 (bearish=-1, bullish=1, neutral≈0)
- "confidence": number between 0 and 1
- "rationale": short explanation in English (1–3 sentences)

Respond with JSON only (no markdown, no backticks).
"""










def classify_headline_sentiment_ollama(
    text: str,
    *,
    model: str = OLLAMA_MODEL,
    max_chars: int = 800,
) -> Dict[str, Any]:
    """
    Classify sentiment of a short headline/blurb using a local Ollama model.

    Parameters
    ----------
    text : str
        Headline or very short summary.
    model : str
        Ollama model name (e.g. 'qwen3:4b').
    max_chars : int
        Max characters to send (just to be safe).

    Returns
    -------
    dict with keys:
        sentiment, score, confidence, rationale, raw
    where 'raw' is the raw model output (for debugging).
    """
    if ollama is None:
        raise RuntimeError("Ollama client not available. Install with `pip install ollama`.")

    snippet = (text or "").strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "... [truncated]"

    user_payload = {
        "headline": snippet,
        "ticker": "AAPL",
    }

    user_text = (
        "Analyze the following headline about Apple Inc. (AAPL).\n\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
        + "\n\nReturn only the JSON object as specified."
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    )
    raw = response["message"]["content"].strip()

    out: Dict[str, Any] = {
        "sentiment": None,
        "score": None,
        "confidence": None,
        "rationale": None,
        "raw": raw,
    }

    # Try robust JSON extraction: find first '{' and last '}'.
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start : end + 1])
        else:
            obj = json.loads(raw)

        out["sentiment"] = obj.get("sentiment")
        out["score"] = obj.get("score")
        out["confidence"] = obj.get("confidence")
        out["rationale"] = obj.get("rationale")
    except Exception as e:
        print(f"[classify_headline_sentiment_ollama] JSON parse error: {e}")

    return out


def score_news_df_with_ollama(
    df: pd.DataFrame,
    *,
    text_col: str = "body",
    model: str = OLLAMA_MODEL,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Apply headline sentiment classification to a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain column `text_col` (e.g. 'body' or 'title').
    text_col : str
        Column used as input text to the model (we'll use 'body' by default).
    model : str
        Ollama model name.
    limit : Optional[int]
        If set, only the first `limit` rows are scored (for quick testing).

    Returns
    -------
    DataFrame : original df with additional columns:
        ['sentiment', 'sentiment_score', 'sentiment_confidence',
         'sentiment_rationale', 'sentiment_raw']
    """
    scored = df.copy()
    n = len(scored)
    if limit is not None:
        n = min(n, limit)

    sentiments: List[Dict[str, Any]] = []
    for i in range(n):
        text = str(scored.iloc[i][text_col])
        print(f"[score_news_df_with_ollama] {i+1}/{n}: {text[:90]}...")
        res = classify_headline_sentiment_ollama(text, model=model)
        sentiments.append(res)

    # For any remaining rows (if limit < len(df)), fill with None
    for _ in range(len(scored) - n):
        sentiments.append(
            {
                "sentiment": None,
                "score": None,
                "confidence": None,
                "rationale": None,
                "raw": None,
            }
        )

    scored["sentiment"] = [s.get("sentiment") for s in sentiments]
    scored["sentiment_score"] = [s.get("score") for s in sentiments]
    scored["sentiment_confidence"] = [s.get("confidence") for s in sentiments]
    scored["sentiment_rationale"] = [s.get("rationale") for s in sentiments]
    scored["sentiment_raw"] = [s.get("raw") for s in sentiments]

    return scored
















@dataclass
class NewsRAGIndex:
    """
    Simple in-memory RAG index for news headlines.

    vectors : (N, d) array of embeddings
    texts   : list of headline/body strings
    meta    : list of per-row dicts (full df row, for debugging / extra context)
    """
    embed_model: str
    vectors: np.ndarray
    texts: List[str]
    meta: List[Dict[str, Any]]


def build_news_rag_index(
    df: pd.DataFrame,
    text_col: str = "body",
    embed_model: str = EMBED_MODEL,
) -> NewsRAGIndex:
    """
    Embed all rows of df[text_col] and return a NewsRAGIndex.

    Parameters
    ----------
    df : DataFrame
        Must have column `text_col` (e.g. 'body' == headline).
    text_col : str
        Column to embed.
    embed_model : str
        Ollama embedding model name, e.g. 'mxbai-embed-large'.

    Returns
    -------
    NewsRAGIndex
    """
    embedder = OllamaEmbeddings(model=embed_model)

    texts = df[text_col].astype(str).tolist()
    print(f"[build_news_rag_index] Embedding {len(texts)} headlines with {embed_model}...")
    vecs_list = embedder.embed_documents(texts)  # List[List[float]]
    vectors = np.asarray(vecs_list, dtype="float32")

    meta = df.to_dict(orient="records")

    return NewsRAGIndex(
        embed_model=embed_model,
        vectors=vectors,
        texts=texts,
        meta=meta,
    )








def retrieve_news_context(
    query_text: str,
    index: NewsRAGIndex,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k similar headlines from the RAG index.

    Returns a list of dicts:
        [{'score': float, 'text': ..., 'meta': {...}}, ...]
    """

    embedder = OllamaEmbeddings(model=index.embed_model)

    q_vec = np.asarray(embedder.embed_query(query_text), dtype="float32")
    V = index.vectors

    # Cosine similarity: (v · q) / (||v|| * ||q||)
    denom = (np.linalg.norm(V, axis=1) * (np.linalg.norm(q_vec) + 1e-8)) + 1e-8
    scores = (V @ q_vec) / denom

    k = min(k, len(V))
    top_idx = np.argsort(-scores)[:k]

    out: List[Dict[str, Any]] = []
    for i in top_idx:
        out.append(
            {
                "score": float(scores[i]),
                "text": index.texts[i],
                "meta": index.meta[i],
            }
        )

    return out






def classify_headline_sentiment_ollama_rag(
    text: str,
    index: Optional[NewsRAGIndex],
    *,
    model: str = OLLAMA_MODEL,
    max_chars: int = 800,
    k: int = 5,
) -> Dict[str, Any]:
    """
    RAG-style sentiment classification:
    - Use similar past headlines as context (from index)
    - Then ask Qwen for sentiment JSON.

    If index is None, this just falls back to the pure classifier.
    """

    if index is None:
        # Fallback: no RAG, use your original function
        return classify_headline_sentiment_ollama(text, model=model, max_chars=max_chars)

    if ollama is None:
        raise RuntimeError("Ollama client not available. Install with `pip install ollama`.")

    snippet = (text or "").strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "... [truncated]"

    # --- Retrieve similar headlines from the index ---
    ctx_docs = retrieve_news_context(snippet, index=index, k=k)

    if ctx_docs:
        ctx_lines = []
        for d in ctx_docs:
            m = d["meta"]
            date = m.get("date_raw", "?")
            src = m.get("source_code", "?")
            ctx_lines.append(
                f"[sim={d['score']:.3f}] {date} {src} – {d['text']}"
            )
        ctx_block = "\n".join(ctx_lines)
    else:
        ctx_block = "(no similar headlines in RAG index)"

    # --- Build user payload with context ---
    user_payload = {
        "headline": snippet,
        "ticker": "AAPL",
        "retrieved_headlines": ctx_lines if ctx_docs else [],
    }

    user_text = (
        "You are a financial news sentiment analyst using RAG.\n"
        "You get a new headline about Apple and a set of similar past headlines.\n\n"
        "Your task: classify sentiment towards AAPL's *future* stock performance.\n\n"
        "New headline:\n"
        f"{snippet}\n\n"
        "Retrieved similar headlines (for context):\n"
        f"{ctx_block}\n\n"
        "Now return a JSON object with keys:\n"
        '  - "sentiment": "bullish" | "bearish" | "neutral"\n'
        '  - "score": float in [-1, 1]\n'
        '  - "confidence": float in [0, 1]\n'
        '  - "rationale": short explanation in English.\n\n'
        "Respond with JSON only, no markdown, no backticks.\n"
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    )
    raw = response["message"]["content"].strip()

    out: Dict[str, Any] = {
        "sentiment": None,
        "score": None,
        "confidence": None,
        "rationale": None,
        "raw": raw,
    }

    # Robust JSON parsing
    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start : end + 1])
        else:
            obj = json.loads(raw)

        out["sentiment"] = obj.get("sentiment")
        out["score"] = obj.get("score")
        out["confidence"] = obj.get("confidence")
        out["rationale"] = obj.get("rationale")
    except Exception as e:
        print(f"[classify_headline_sentiment_ollama_rag] JSON parse error: {e}")

    return out


def score_news_df_with_ollama_rag(
    df: pd.DataFrame,
    index: Optional[NewsRAGIndex],
    *,
    text_col: str = "body",
    model: str = OLLAMA_MODEL,
    limit: Optional[int] = None,
    k: int = 5,
) -> pd.DataFrame:
    """
    Apply RAG-based headline sentiment classification to a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain column `text_col`.
    index : NewsRAGIndex | None
        RAG index built from the same or larger corpus.
    text_col : str
        Column used as input text.
    model : str
        Ollama model name (e.g. 'qwen3:4b').
    limit : Optional[int]
        If set, only the first `limit` rows are scored.
    k : int
        Number of retrieved context headlines for each classification.

    Returns
    -------
    DataFrame with additional columns:
        sentiment, sentiment_score, sentiment_confidence,
        sentiment_rationale, sentiment_raw
    """
    scored = df.copy()
    n = len(scored)
    if limit is not None:
        n = min(n, limit)

    sentiments: List[Dict[str, Any]] = []
    for i in range(n):
        text = str(scored.iloc[i][text_col])
        print(f"[score_news_df_with_ollama_rag] {i+1}/{n}: {text[:90]}...")
        res = classify_headline_sentiment_ollama_rag(
            text,
            index=index,
            model=model,
            k=k,
        )
        sentiments.append(res)

    # Fill remaining rows (if limit < len(df)) with None
    for _ in range(len(scored) - n):
        sentiments.append(
            {
                "sentiment": None,
                "score": None,
                "confidence": None,
                "rationale": None,
                "raw": None,
            }
        )

    scored["sentiment"] = [s.get("sentiment") for s in sentiments]
    scored["sentiment_score"] = [s.get("score") for s in sentiments]
    scored["sentiment_confidence"] = [s.get("confidence") for s in sentiments]
    scored["sentiment_rationale"] = [s.get("rationale") for s in sentiments]
    scored["sentiment_raw"] = [s.get("raw") for s in sentiments]

    return scored
