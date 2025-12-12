import json
from typing import Any, Dict, Optional
from typing import List, Dict, Optional
import pandas as pd

try:
    import ollama  # pip install ollama
except ImportError:
    ollama = None
    print("Warning: 'ollama' is not installed. Run `pip install ollama` to use the sentiment functions.")


# Default model; override via env OLLAMA_MODEL if you like
import os


def classify_headline_sentiment_ollama(
    text: str,
    model = os.getenv("OLLAMA_MODEL", "gemma3:1b"),
    ticker = "TSLA",
    name = "Tesla Inc.",
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
    

    SENTIMENT_SYSTEM_PROMPT = """\
    You are a financial news sentiment analyst focusing on a single stock.

    Given a short headline or short blurb about {name}. (ticker: {ticker}),
    classify the sentiment with respect to {name}'s FUTURE stock performance as:

    - "bullish": mainly positive impact on {ticker}
    - "bearish": mainly negative impact on {ticker}
    - "neutral": mixed or unclear impact, or {name} only mentioned tangentially

    Return a compact JSON object with keys:
    - "sentiment": one of ["bullish", "bearish", "neutral"]
    - "score": a number between -1 and 1 (bearish=-1, bullish=1, neutral≈0)
    - "confidence": number between 0 and 1
    - "rationale": short explanation in English (1–3 sentences)

    Respond with JSON only (no markdown, no backticks).
    """


    snippet = (text or "").strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "... [truncated]"

    user_payload = {
        "headline": snippet,
        "ticker": ticker,
    }

    user_text = (
        "Analyze the following headline about {name} ({ticker}).\n\n"
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
    model = os.getenv("OLLAMA_MODEL", "gemma3:1b"),
    ticker = "TSLA",
    name = "Tesla Inc.",
    *,
    text_col: str = "body",
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
        res = classify_headline_sentiment_ollama(text, model=model, ticker=ticker, name=name)
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
