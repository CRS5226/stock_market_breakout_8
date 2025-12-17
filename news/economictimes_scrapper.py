#!/usr/bin/env python3
"""
economic_times_scraper.py
-------------------------------------------------
Scrapes Economic Times for:
1️⃣ General / Non-finance news:
    https://economictimes.indiatimes.com/news/newsblogs
2️⃣ Market / Finance news:
    https://economictimes.indiatimes.com/markets
3️⃣ Expert Views:
    https://economictimes.indiatimes.com/markets/expert-views

Rules:
- Non-finance: keep existing summary, no synopsis scraping.
- Finance: fetch synopsis or article content, saved as 'summary'.
- Expert views: keep page summary; if missing, fetch synopsis/content.

Outputs:
- scrapper_data/et_nonfinance_news_<timestamp>.json
- scrapper_data/et_market_news_<timestamp>.json
- scrapper_data/et_expert_views_<timestamp>.json

Author: GPT-5 Assistant
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import json
import os
import time

BASE_URL = "https://economictimes.indiatimes.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ET-Scraper/1.2)"}

# Ensure output directory exists
os.makedirs("scrapper_data", exist_ok=True)


# ==========================================================
# Utility Functions
# ==========================================================


def clean_text(text: str) -> str:
    """Normalize and clean text."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def save_json(data, filename_prefix: str):
    """Save data as timestamped JSON file in scrapper_data/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"scrapper_data/{filename_prefix}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[💾] Saved {len(data)} items → {out_path}")
    return out_path


def fetch_article_content(url: str) -> str:
    """
    Fetch the article synopsis (preferred) or main <p> body text if synopsis not found.
    Used for finance and expert views only.
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        # Try to find Synopsis section
        synopsis_tag = soup.select_one("div.artSyn.bgPink p.summary")
        if synopsis_tag:
            return clean_text(synopsis_tag.get_text())

        # Fallback to main article <p> tags
        content_paragraphs = soup.select("div.artText p")
        if content_paragraphs:
            body_text = " ".join(clean_text(p.get_text()) for p in content_paragraphs)
            return body_text[:2000]  # trim overly long text
    except Exception as e:
        print(f"[⚠️] Failed to fetch content from {url}: {e}")
    return None


# ==========================================================
# 📰 Non-Finance / General News Scraper
# ==========================================================


def fetch_et_general_news(limit: int = 20):
    """Fetches latest non-finance articles from ET /news/newsblogs."""
    target_url = f"{BASE_URL}/news/newsblogs"
    print(f"[🗞️] Fetching Non-Finance ET News → {target_url}")

    try:
        r = requests.get(target_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[⚠️] Failed to fetch ET general page: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    stories = soup.select("div.eachStory")

    print(f"[📊] Found {len(stories)} general stories on page.")
    articles = []

    for story in stories[:limit]:
        try:
            title_tag = story.find("h3")
            title = clean_text(title_tag.get_text()) if title_tag else "Untitled"

            link_tag = story.find("a", href=True)
            url = link_tag["href"]
            if url.startswith("/"):
                url = BASE_URL + url

            time_tag = story.find("time", {"class": "date-format"})
            published_at = (
                time_tag["data-time"]
                if time_tag and time_tag.has_attr("data-time")
                else "Unknown"
            )

            summary_tag = story.find("p", {"class": "line_3"})
            summary = clean_text(summary_tag.get_text()) if summary_tag else ""

            img_tag = story.find("img")
            img_url = img_tag["src"] if img_tag else None

            articles.append(
                {
                    "title": title,
                    "url": url,
                    "summary": summary,
                    "published_at": published_at,
                    "image": img_url,
                    "category": "non-finance",
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                }
            )
        except Exception as e:
            print(f"[⚠️] Skipped a general story due to error: {e}")
            continue

    print(f"[✅] Extracted {len(articles)} non-finance articles.")
    return articles


# ==========================================================
# 💹 Market / Finance News Scraper
# ==========================================================


def fetch_et_market_news(limit: int = 20):
    """Fetches latest market/finance news from ET Markets."""
    target_url = f"{BASE_URL}/markets"
    print(f"[💹] Fetching Market ET News → {target_url}")

    try:
        r = requests.get(target_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[⚠️] Failed to fetch ET markets page: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    stories_section = soup.select_one("div.FirstFoldWidget_topStories__tBR9H")
    if not stories_section:
        print("[⚠️] Could not find top stories section.")
        return []

    items = stories_section.select("li.FirstFoldWidget_stry__Sr2wu a[href]")
    print(f"[📊] Found {len(items)} market articles in top stories.")

    articles = []
    for idx, a in enumerate(items[:limit], start=1):
        try:
            url = a["href"]
            if url.startswith("/"):
                url = BASE_URL + url
            title = clean_text(a.get_text())
            img_tag = a.find("img")
            img_url = img_tag["src"] if img_tag else None

            # Fetch synopsis/content and rename to summary
            summary = fetch_article_content(url)
            if summary:
                print(f"[📝] ({idx}) Summary fetched for: {title[:60]}...")
            else:
                print(f"[⚠️] ({idx}) No summary found for: {title[:60]}...")

            articles.append(
                {
                    "title": title,
                    "url": url,
                    "summary": summary,
                    "image": img_url,
                    "category": "finance",
                    "published_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                }
            )
            time.sleep(0.5)
        except Exception as e:
            print(f"[⚠️] Skipped a market story due to error: {e}")
            continue

    print(f"[✅] Extracted {len(articles)} finance articles with summaries.")
    return articles


# ==========================================================
# 🧠 Expert Views Scraper
# ==========================================================


def fetch_et_expert_views(limit: int = 20):
    """Fetches expert views and their synopsis if summary missing."""
    target_url = f"{BASE_URL}/markets/expert-views"
    print(f"[🧠] Fetching Expert Views → {target_url}")

    try:
        r = requests.get(target_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except Exception as e:
        print(f"[⚠️] Failed to fetch Expert Views page: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    stories = soup.select("div.eachStory")

    print(f"[📊] Found {len(stories)} expert view stories.")
    articles = []

    for idx, story in enumerate(stories[:limit], start=1):
        try:
            title_tag = story.find("h3")
            title = clean_text(title_tag.get_text()) if title_tag else "Untitled"

            link_tag = story.find("a", href=True)
            url = link_tag["href"]
            if url.startswith("/"):
                url = BASE_URL + url

            time_tag = story.find("time", {"class": "date-format"})
            published_at = (
                time_tag["data-time"]
                if time_tag and time_tag.has_attr("data-time")
                else "Unknown"
            )

            summary_tag = story.find("p")
            summary = clean_text(summary_tag.get_text()) if summary_tag else None

            img_tag = story.find("img")
            img_url = img_tag["src"] if img_tag else None

            # Stock info (optional)
            stock_widget = story.select_one("div.stockDataWidget")
            stock_data = clean_text(stock_widget.get_text()) if stock_widget else None

            # Only fetch from URL if summary missing
            if not summary:
                summary = fetch_article_content(url)
                if summary:
                    print(f"[📝] ({idx}) Added summary from URL for: {title[:60]}...")
                else:
                    print(f"[⚠️] ({idx}) No summary found for: {title[:60]}...")
            else:
                print(f"[✅] ({idx}) Summary already present for: {title[:60]}")

            articles.append(
                {
                    "title": title,
                    "url": url,
                    "summary": summary,
                    "published_at": published_at,
                    "image": img_url,
                    "stock_data": stock_data,
                    "category": "expert-view",
                    "scraped_at": datetime.utcnow().isoformat() + "Z",
                }
            )
            time.sleep(0.5)
        except Exception as e:
            print(f"[⚠️] Skipped an expert view due to error: {e}")
            continue

    print(f"[✅] Extracted {len(articles)} expert views with valid summaries.")
    return articles


# ==========================================================
# 🗂️ Main Execution
# ==========================================================


def main():
    general_news = fetch_et_general_news(limit=15)
    market_news = fetch_et_market_news(limit=15)
    expert_views = fetch_et_expert_views(limit=15)

    save_json(general_news, "et_nonfinance_news")
    save_json(market_news, "et_market_news")
    save_json(expert_views, "et_expert_views")

    print("[🏁] Economic Times Scraping Completed Successfully.")


if __name__ == "__main__":
    main()
