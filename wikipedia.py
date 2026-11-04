"""Search and read English Wikipedia through the public MediaWiki API. No login needed.

Every failure returns a specific message with a hint, because the model cannot see an
exception and a generic error gives it nothing to correct.
"""

import html
import json
import re
from urllib.parse import unquote, urlparse

import httpx

API_URL = "https://en.wikipedia.org/w/api.php"
# Wikimedia's policy asks clients to identify themselves. A bare name can get a 403.
HEADERS = {"User-Agent": "mlb-deep-research/0.1 (https://github.com/mvnye/MLB-Deep-Research)",
           "Accept": "application/json"}
TIMEOUT_SECONDS = 20
# Enough to work with. The whole of a long season article would crowd the context.
MAX_CHARS = 12_000
MAX_RESULTS = 10


def _error(message: str, hint: str) -> str:
    return json.dumps({"error": message, "hint": hint})


def _get(params: dict) -> dict:
    response = httpx.get(API_URL, params={**params, "format": "json", "formatversion": 2},
                         headers=HEADERS, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _request_error(e: httpx.HTTPError) -> str:
    if isinstance(e, httpx.TimeoutException):
        return _error(f"Wikipedia did not answer within {TIMEOUT_SECONDS} seconds.", "Try once more.")
    if isinstance(e, httpx.HTTPStatusError):
        return _error(f"Wikipedia returned HTTP {e.response.status_code}.",
                      "Wait, then try once more. If it fails again, say the context is unavailable.")
    return _error(f"Could not reach Wikipedia: {type(e).__name__}.",
                  "Try once more. If it fails again, say the context is unavailable.")


def title_from_url(url: str) -> str:
    """'https://en.wikipedia.org/wiki/New_York_Mets#History' -> 'New York Mets'.

    A bare title passes through.
    """
    path = urlparse(url).path
    title = path.split("/wiki/", 1)[1] if "/wiki/" in path else url
    return unquote(title).replace("_", " ").strip()


def article_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def search(query: str, limit: int = 5) -> str:
    """Search article titles and text. Returns JSON with title, url and snippet per hit."""
    if not query.strip():
        return _error("The search query is empty.", "Name a team and season, e.g. '2025 New York Mets season'.")
    try:
        data = _get({"action": "query", "list": "search", "srsearch": query, "srinfo": "suggestion",
                     "srlimit": max(1, min(limit, MAX_RESULTS))})
    except httpx.HTTPError as e:
        return _request_error(e)

    hits = data.get("query", {}).get("search", [])
    if not hits:
        suggestion = data.get("query", {}).get("searchinfo", {}).get("suggestion")
        example = "a different query, such as the full team name and season: '2025 New York Mets season'"
        hint = f"Try {suggestion!r} or {example}." if suggestion else f"Try {example}."
        return _error(f"Wikipedia has no results for {query!r}.", hint)
    return json.dumps({"results": [{
        "title": h["title"], "url": article_url(h["title"]),
        "snippet": html.unescape(re.sub(r"<[^>]+>", "", h.get("snippet", ""))),
    } for h in hits]})


def read(title: str, offset: int = 0) -> str:
    """Read an article's plain text from offset. Returns JSON with title, url and text."""
    name = title_from_url(title)
    if not name:
        return _error("The article title is empty.", "Pass a title or URL from search results.")
    try:
        data = _get({"action": "query", "prop": "extracts|pageprops", "ppprop": "disambiguation",
                     "explaintext": 1, "exsectionformat": "wiki", "redirects": 1, "titles": name})
    except httpx.HTTPError as e:
        return _request_error(e)

    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing") or pages[0].get("invalid"):
        return _error(f"There is no Wikipedia article titled {name!r}.",
                      "Call the search tool and use a title from its results.")
    page = pages[0]
    if "disambiguation" in page.get("pageprops", {}):
        return _error(f"{page['title']!r} is a disambiguation page, not an article.",
                      "Search for a more specific title, such as the team and season.")

    text = page.get("extract", "")
    if not text:
        return _error(f"The article {page['title']!r} has no readable text.", "Try another article.")
    start = max(offset, 0)
    if start >= len(text):
        return _error(f"Offset {start} is past the end of the article ({len(text)} characters).",
                      "Use a smaller offset.")
    end = start + MAX_CHARS
    result = {"title": page["title"], "url": article_url(page["title"]), "text": text[start:end]}
    if end < len(text):
        result["truncated"] = (f"Showing characters {start} to {end} of {len(text)}. "
                               f"Call again with offset={end} to read on.")
    return json.dumps(result)


if __name__ == "__main__":
    # Smoke test: uv run wikipedia.py
    print(search("2025 New York Mets season", 2)[:400])
    print(read("2025 New York Mets season")[:400])
