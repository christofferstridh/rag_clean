import os
import re
import time
import requests
from pyprojroot import here


def sanitize_filename(filename):
    return re.sub(r"[^a-zA-Z0-9]+", "_", filename).strip("_")


def _request_json(url, params, session, max_retries=5):
    user_agent = "RagCorpusBot/1.0 (https://example.org/rag-corpus; bot@example.org)"
    headers = {
        "User-Agent": user_agent,
        "Api-User-Agent": user_agent,
    }
    delay = 1.0

    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, headers=headers, timeout=30)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = (
                    float(retry_after) if retry_after and retry_after.isdigit() else delay
                )
                print(f"Rate limited by Wikipedia. Waiting {wait_seconds}s before retrying...")
                time.sleep(wait_seconds)
                delay *= 2
                continue

            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Wikipedia request failed: {exc}") from exc
            print(f"Request failed ({exc}). Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("Wikipedia request failed after retries")


def search_wikipedia_titles(search_term, results=10, session=None):
    params = {
        "action": "opensearch",
        "search": search_term,
        "limit": results,
        "namespace": "0",
        "format": "json",
        "redirects": "resolve",
    }
    data = _request_json(
        "https://en.wikipedia.org/w/api.php", params, session or requests.Session()
    )
    return data[1] if len(data) > 1 else []


def fetch_page_content(title, session=None):
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
        "redirects": 1,
    }
    data = _request_json(
        "https://en.wikipedia.org/w/api.php", params, session or requests.Session()
    )
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise ValueError(f"No page data returned for '{title}'")

    page = next(iter(pages.values()))
    content = page.get("extract", "").strip()
    if not content:
        raise ValueError(f"No extract found for '{title}'")
    return content


def generate_corpus(search_term="human rights", num_articles=50, output_dir="all_articles"):
    os.makedirs(here() / "resources" / output_dir, exist_ok=True)

    articles = []
    request_limit = max(1, min(int(num_articles), 50))
    session = requests.Session()

    try:
        search_results = search_wikipedia_titles(
            search_term, results=request_limit, session=session
        )
    except Exception as e:
        print(f"Unable to search Wikipedia: {e}")
        return articles

    for i, title in enumerate(search_results, 1):
        try:
            time.sleep(1.0)
            content = fetch_page_content(title, session=session)
            filename = f"{sanitize_filename(title)}.txt"
            filepath = here() / "resources" / output_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            articles.append((title, filepath))
            print(f"{i}/{len(search_results)} -> {title}")
        except Exception as e:
            print(f"Error processing '{title}': {str(e)}")
            continue

    print(f"Saved {len(articles)} articles to {output_dir}")
    return articles


if __name__ == "__main__":
    generate_corpus()
