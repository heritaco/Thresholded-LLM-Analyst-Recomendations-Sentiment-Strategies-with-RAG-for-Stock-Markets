import time
import random
from typing import List, Dict, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd


def make_session(
    user_agent: Optional[str] = None,
    timeout: float = 15.0,
) -> requests.Session:
    """
    Create a configured requests Session with a realistic User-Agent.
    """
    if user_agent is None:
        user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        )

    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": user_agent,
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Referer": "https://www.google.com/",
        }
    )
    sess.timeout = timeout
    return sess


def fetch_html(
    url: str,
    session: Optional[requests.Session] = None,
    sleep: float = 1.5,
) -> str:
    """
    Fetch a single HTML page. Raises HTTPError if status != 200.
    Respects a small sleep to avoid hammering the site.
    """
    sess = session or make_session()
    resp = sess.get(url)
    resp.raise_for_status()  # if 403/404/etc., this will raise
    if sleep:
        time.sleep(sleep)
    return resp.text


def parse_marketscreener_analyst_reco(
    html: str,
    base_url: str = "https://www.marketscreener.com",
) -> List[Dict[str, str]]:
    """
    Parse the 'Analyst Reco.' block from a MarketScreener stock page.

    The result is a list of dicts with at least:
      - title: headline text (e.g. 'APPLE INC : UBS reiterates its Neutral rating')
      - url:  absolute URL to the article page
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) Find the h3 that labels the block
    header = soup.find("h3", string=lambda s: s and "Analyst Reco" in s)
    if not header:
        return []

    articles: List[Dict[str, str]] = []

    # 2) Walk forward until the next h3 (next major section)
    for tag in header.find_all_next():
        # Stop when we reach the next section header (e.g. "Filters", "News", etc.)
        if tag.name == "h3" and tag is not header:
            break

        if tag.name == "a" and tag.get("href"):
            title = tag.get_text(strip=True)
            # Heuristic: keep links that look like broker-research headlines.
            # You can relax or generalize this later.
            if not title:
                continue
            if "APPLE" not in title.upper():
                # Avoid navigation / unrelated links
                continue
            href = tag["href"]
            url = urljoin(base_url, href)

            # Deduplicate by URL
            if any(a["url"] == url for a in articles):
                continue

            articles.append(
                {
                    "title": title,
                    "url": url,
                }
            )

    return articles




def extract_article_text(html: str) -> str:
    """
    Very simple article text extractor.

    Strategy:
      1. If there's an <article> tag, use all <p> inside it.
      2. Else, fall back to all <p> on the page.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Prefer an <article> container if present
    article_tag = soup.find("article")
    if article_tag:
        paragraphs = article_tag.find_all("p")
    else:
        paragraphs = soup.find_all("p")

    texts = [
        p.get_text(" ", strip=True)
        for p in paragraphs
        if p.get_text(strip=True)
    ]
    return "\n\n".join(texts)


def enrich_articles_with_body(
    articles: List[Dict[str, str]],
    session: Optional[requests.Session] = None,
    sleep: float = 1.5,
    max_articles: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Fetch each article URL and add 'body' to every item.

    If max_articles is set, only the first N articles are fetched.
    """
    sess = session or make_session()

    if max_articles is not None:
        subset = articles[: max_articles]
    else:
        subset = articles

    out: List[Dict[str, str]] = []
    for art in subset:
        url = art["url"]
        try:
            html = fetch_html(url, session=sess, sleep=sleep)
            body = extract_article_text(html)
        except Exception as e:
            body = f"[ERROR: {e}]"

        enriched = dict(art)
        enriched["body"] = body
        out.append(enriched)

    # If we limited to max_articles, we might want to keep the others as well
    if max_articles is not None and max_articles < len(articles):
        for art in articles[max_articles:]:
            still = dict(art)
            still["body"] = ""
            out.append(still)

    return out



def scrape(
    base_url: str = "https://www.marketscreener.com/quote/stock/APPLE-INC-4849/news-broker-research/",
    fetch_bodies: bool = True,
    sleep: float = 1.5,
    max_articles_body: Optional[int] = 20,
) -> pd.DataFrame:
    """
    End-to-end scraper for the MarketScreener 'Analyst Reco.' panel.

    Parameters
    ----------
    base_url : str
        URL of the MarketScreener broker-research page for a stock.
    fetch_bodies : bool
        If True, fetch each article and extract a text body.
        If False, only headlines + URLs are returned.
    sleep : float
        Delay between HTTP requests in seconds (politeness).
    max_articles_body : Optional[int]
        Max number of articles for which we fetch full HTML and body.
        None = fetch body for all.

    Returns
    -------
    df : pandas.DataFrame
        Columns at least: ['title', 'url'] and optionally 'body'.
    """
    sess = make_session()

    print(f"[scrape_marketscreener] Fetching listing page: {base_url}")
    html = fetch_html(base_url, session=sess, sleep=sleep)
    articles = parse_marketscreener_analyst_reco(html, base_url=base_url)

    if not articles:
        print("[scrape_marketscreener] No articles parsed from 'Analyst Reco.' section.")
        return pd.DataFrame(columns=["title", "url", "body"])

    print(f"[scrape_marketscreener] Parsed {len(articles)} headlines.")

    if fetch_bodies:
        print("[scrape_marketscreener] Fetching article bodies...")
        articles = enrich_articles_with_body(
            articles,
            session=sess,
            sleep=sleep,
            max_articles=max_articles_body,
        )

    df = pd.DataFrame(articles)
    return df
