#!/usr/bin/env python3
"""
llm_news_sentiment.py
-------------------------------------------------
Economic Times scraper + GPT-5-mini sentiment analyzer (real-time version)

1️⃣ Scrapes:
   - /news/newsblogs         → Non-finance news
   - /markets                → Finance news
   - /markets/expert-views   → Expert opinions
2️⃣ Runs GPT-5-mini + embeddings RAG classifier
3️⃣ Maintains a single live JSON file (news/data/news.json)
4️⃣ Designed for 5-minute updates via server loop

Author: GPT-5 Assistant
"""

import os, re, json, time, pytz, requests, pickle
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from dateutil import parser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# === Load Keys & Config ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ Missing OPENAI_API_KEY in .env")

IST = pytz.timezone("Asia/Kolkata")
MODEL = "gpt-5-mini"
EMBED_MODEL = "text-embedding-3-large"
SYMBOL_SIM_THRESHOLD = 0.3

# === Paths ===
EMB_PATH = "./news/data/company_embeddings.pkl"
DATA_DIR = os.path.join("news", "data")
FINAL_JSON_PATH = os.path.join(DATA_DIR, "news.json")
VECTOR_DIR = os.path.join(DATA_DIR, "vector_store")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# === Load Company Embeddings ===
if not os.path.exists(EMB_PATH):
    raise FileNotFoundError("❌ Missing company_embeddings.pkl")
with open(EMB_PATH, "rb") as f:
    company_embeddings = pickle.load(f)
print(f"[📊] Loaded {len(company_embeddings)} company embeddings")

# === Vectorstore Setup ===
embedding_fn = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(
    collection_name="news_articles",
    embedding_function=embedding_fn,
    persist_directory=VECTOR_DIR,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# === Utility ===
def now_ts():
    return datetime.now(IST).strftime("%Y-%m-%d_%H-%M-%S")


def clean_text(t):
    return re.sub(r"\s+", " ", t.strip()) if t else ""


def compute_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    v1, v2 = np.array(v1).reshape(1, -1), np.array(v2).reshape(1, -1)
    if v1.shape[1] != v2.shape[1]:
        m = min(v1.shape[1], v2.shape[1])
        v1, v2 = v1[:, :m], v2[:, :m]
    return float(cosine_similarity(v1, v2)[0][0])


def parse_any_datetime(date_str):
    """Handles multiple date formats safely and converts to IST timezone."""
    if not date_str or "Unknown" in date_str:
        return None
    try:
        tzinfos = {"IST": pytz.timezone("Asia/Kolkata")}
        dt = parser.parse(date_str, fuzzy=True, tzinfos=tzinfos)
        if dt.tzinfo is None:
            dt = IST.localize(dt)
        else:
            dt = dt.astimezone(IST)
        return dt
    except Exception:
        return None


# ==========================================================
# 🌐 Economic Times Scraper
# ==========================================================
BASE_URL = "https://economictimes.indiatimes.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (ET-Scraper/2.0)"}


def fetch_article_content(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        syn = soup.select_one("div.artSyn.bgPink p.summary")
        if syn:
            return clean_text(syn.get_text())
        paras = soup.select("div.artText p")
        if paras:
            return " ".join(clean_text(p.get_text()) for p in paras)[:2000]
    except Exception as e:
        print(f"[⚠️] Content fetch failed {url[:60]} → {e}")
    return None


def fetch_et_general_news(limit=5):
    print(f"\n[🗞️] Fetching Non-Finance ET News…")
    try:
        r = requests.get(f"{BASE_URL}/news/newsblogs", headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        stories = soup.select("div.eachStory")[:limit]
        data = []
        for s in stories:
            try:
                url = BASE_URL + s.a["href"]
                data.append(
                    {
                        "title": clean_text(s.h3.get_text()),
                        "url": url,
                        "summary": clean_text(s.select_one("p.line_3").get_text()),
                        "published_at": s.time["data-time"] if s.time else "Unknown",
                        "category": "non-finance",
                    }
                )
            except Exception:
                continue
        print(f"[✅] {len(data)} Non-finance fetched")
        return data
    except Exception as e:
        print(f"[⚠️] Failed to fetch general news → {e}")
        return []


def fetch_et_market_news(limit=5):
    print(f"\n[💹] Fetching Finance ET News…")
    try:
        r = requests.get(f"{BASE_URL}/markets", headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        stories = soup.select("li.FirstFoldWidget_stry__Sr2wu a[href]")[:limit]
        data = []
        for s in stories:
            try:
                url = BASE_URL + s["href"] if s["href"].startswith("/") else s["href"]
                title = clean_text(s.get_text())
                summary = fetch_article_content(url)
                data.append(
                    {
                        "title": title,
                        "url": url,
                        "summary": summary,
                        "category": "finance",
                        "published_at": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
            except Exception:
                continue
        print(f"[✅] {len(data)} Finance fetched")
        return data
    except Exception as e:
        print(f"[⚠️] Failed to fetch finance news → {e}")
        return []


def fetch_et_expert_views(limit=5):
    print(f"\n[🧠] Fetching Expert Views…")
    try:
        r = requests.get(
            f"{BASE_URL}/markets/expert-views", headers=HEADERS, timeout=15
        )
        soup = BeautifulSoup(r.text, "html.parser")
        stories = soup.select("div.eachStory")[:limit]
        data = []
        for s in stories:
            try:
                url = BASE_URL + s.a["href"]
                summary_tag = s.find("p")
                summary = clean_text(summary_tag.get_text()) if summary_tag else None
                if not summary:
                    summary = fetch_article_content(url)
                stock_widget = s.select_one("div.stockDataWidget")
                stock_data = (
                    clean_text(stock_widget.get_text()) if stock_widget else None
                )
                data.append(
                    {
                        "title": clean_text(s.h3.get_text()),
                        "url": url,
                        "summary": summary,
                        "stock_data": stock_data,
                        "published_at": s.time["data-time"] if s.time else "Unknown",
                        "category": "expert-view",
                    }
                )
            except Exception:
                continue
        print(f"[✅] {len(data)} Expert views fetched")
        return data
    except Exception as e:
        print(f"[⚠️] Failed to fetch expert views → {e}")
        return []


# ==========================================================
# 🧩 GPT-5 Mini Sentiment Analyzer
# ==========================================================
prompt_template = """
You are an Indian stock analyst.
Analyze the article below and decide for each stock among [{symbols}]
if the sentiment implies "BUY" or "No Action".
Output JSON only.

Context (past news):
{context}

Title: {title}
Published At: {published_at}
URL: {url}

Article:
{content}
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["symbols", "context", "title", "url", "content", "published_at"],
)
llm = ChatOpenAI(model_name=MODEL, temperature=0.3, openai_api_key=OPENAI_API_KEY)


def find_relevant_companies(news_text, top_n=5):
    """
    Uses cosine similarity between news text embeddings and stored company embeddings
    to find the top N relevant stock symbols.
    """
    try:
        vec = np.array(embedding_fn.embed_query(news_text))
        sims = []
        for sym, emb in company_embeddings.items():
            sc = compute_similarity(vec, emb)
            sims.append((sym, sc))
        sims.sort(key=lambda x: x[1], reverse=True)

        top_matches = [(s, sc) for s, sc in sims[:top_n] if sc >= SYMBOL_SIM_THRESHOLD]
        if not top_matches:
            return []
        print(
            f"[🔎] Relevant symbols → {', '.join([f'{s}({sc:.2f})' for s, sc in top_matches])}"
        )
        return [s for s, _ in top_matches]
    except Exception as e:
        print(f"[⚠️] Embedding similarity computation failed: {e}")
        return []


def analyze_article(article):
    """Analyze a single article with GPT and embedding-based company mapping."""
    title, url = article["title"], article["url"]
    content = article.get("summary") or fetch_article_content(url)
    published_at = article.get("published_at", "Unknown")

    if not content or len(content) < 80:
        print(f"[🚫] Skipping short: {title[:60]}")
        return {}, []

    # 🧩 Adaptive similarity threshold based on category
    if article.get("category") == "non-finance":
        local_threshold = 0.18
    else:
        local_threshold = 0.3

    global SYMBOL_SIM_THRESHOLD
    SYMBOL_SIM_THRESHOLD = local_threshold

    # 🧩 Find relevant companies using cosine similarity
    candidates = find_relevant_companies(content, top_n=7)
    if not candidates:
        print(f"[⚠️] No symbols for {title[:60]}")
        return {}, []

    print(f"[🔎] Relevant symbols → {', '.join(candidates)}")

    # 🔍 Retrieve contextual docs from Chroma vectorstore
    try:
        retrieved_docs = retriever.invoke(content)
        context_text = (
            "\n\n".join([d.page_content for d in retrieved_docs])
            if retrieved_docs
            else ""
        )
    except Exception as e:
        print(f"[⚠️] Retriever error: {e}")
        context_text = ""

    # 🧠 Build GPT prompt
    prompt_filled = PROMPT.format(
        symbols=", ".join(candidates),
        context=context_text,
        title=title,
        url=url,
        content=content,
        published_at=published_at,
    )

    # ============================================================
    # 📝 DEBUG LOG: Save GPT prompt for inspection + token count
    # ============================================================
    try:
        import tiktoken

        log_dir = "debug_prompts_news"
        os.makedirs(log_dir, exist_ok=True)

        # Create safe file name (truncate if too long)
        safe_title = re.sub(r"[^A-Za-z0-9]+", "_", title)[:60]
        ts_str = datetime.now(IST).strftime("%Y-%m-%d_%H-%M-%S")
        log_file = os.path.join(log_dir, f"{safe_title}_{ts_str}.txt")

        # Estimate token count
        try:
            encoding = tiktoken.encoding_for_model(MODEL)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(prompt_filled))

        # Save to file
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(prompt_filled)
            f.write("\n\n# ================================\n")
            f.write(f"# PROMPT TOKEN SIZE: {token_count}\n")
            f.write("# ================================\n")

        print(f"[🧾 Prompt Saved] {log_file} | INPUT TOKENS: {token_count}")

        # Optional: warn if prompt too large
        if token_count > 14000:
            print(f"[⚠️ LARGE PROMPT WARNING] {safe_title} → {token_count} tokens!")

    except Exception as e:
        print(f"[⚠️ Failed to log GPT prompt for {title[:50]}: {e}]")


    # ⚙️ Run GPT for sentiment classification
    try:
        raw = llm.invoke(prompt_filled)
        match = re.search(r"\{.*\}", raw.content, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            print(f"[⚠️] No valid JSON in GPT output for {title[:60]}")
            parsed = {sym: "No Action" for sym in candidates}
    except Exception as e:
        print(f"[⚠️] GPT parse error for {title[:60]} → {e}")
        parsed = {sym: "No Action" for sym in candidates}

    # 💾 Add to Chroma for context learning
    try:
        doc = Document(page_content=content, metadata={"title": title, "url": url})
        vectorstore.add_documents([doc])  # ✅ No .persist() call
        print(f"[💾] Vector added for: {title[:60]}")
    except Exception as e:
        print(f"[⚠️] Chroma add_documents error: {e}")

    print(f"[✅] {title[:60]} ({published_at}) → {parsed}")
    return parsed, candidates


# ==========================================================
# 🧾 JSON Handling
# ==========================================================
def load_existing_json():
    """Safely load JSON or reset if empty/corrupt."""
    if not os.path.exists(FINAL_JSON_PATH):
        return {"last_updated": "1970-01-01_00-00-00", "articles": []}
    try:
        with open(FINAL_JSON_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {"last_updated": "1970-01-01_00-00-00", "articles": []}
            return json.loads(content)
    except (json.JSONDecodeError, OSError):
        print(f"[⚠️] Resetting empty or corrupt news.json.")
        return {"last_updated": "1970-01-01_00-00-00", "articles": []}


def save_latest_json(data):
    with open(FINAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[💾] Updated → {FINAL_JSON_PATH}")


# ==========================================================
# 🚀 update_news_sentiment() — Main Logic
# ==========================================================
def update_news_sentiment():
    print("\n[⚙️] Running update_news_sentiment()")

    existing_data = load_existing_json()
    last_update_str = existing_data.get("last_updated", "1970-01-01_00-00-00")

    try:
        last_update_dt = datetime.strptime(last_update_str, "%Y-%m-%d_%H-%M-%S")
        if last_update_dt.tzinfo is None:
            last_update_dt = IST.localize(last_update_dt)
    except Exception:
        last_update_dt = IST.localize(datetime(1970, 1, 1))

    is_first_run = last_update_str.startswith("1970")
    print(f"[🔍] is_first_run={is_first_run}, last_update={last_update_dt}")

    # Fetch all categories
    gen = fetch_et_general_news(limit=5)
    fin = fetch_et_market_news(limit=5)
    exp = fetch_et_expert_views(limit=5)
    all_articles = gen + fin + exp

    new_articles = []
    for art in all_articles:
        pub_dt = parse_any_datetime(art.get("published_at"))
        if not pub_dt:
            # fallback to "now" for finance/expert where time may be missing
            pub_dt = datetime.now(IST)
        if is_first_run or pub_dt > last_update_dt:
            new_articles.append(art)

    # Deduplicate by URL
    existing_urls = {a["url"] for a in existing_data.get("articles", [])}
    new_articles = [a for a in new_articles if a["url"] not in existing_urls]

    print(f"[🆕] Found {len(new_articles)} new unique articles after {last_update_str}")

    if not new_articles:
        print("[ℹ️] No new news. Skipping GPT run.\n")
        return

    for art in new_articles:
        try:
            res, cands = analyze_article(art)
            art["result"] = res
            art["candidates"] = cands
            time.sleep(0.5)
        except Exception as e:
            print(f"[⚠️] Error analyzing article: {e}")
            art["result"] = {}
            art["candidates"] = []

    existing_data["articles"].extend(new_articles)
    existing_data["last_updated"] = now_ts()
    save_latest_json(existing_data)
    print(f"[🏁] update_news_sentiment() completed.\n")


# ==========================================================
# Optional — manual run
# ==========================================================
if __name__ == "__main__":
    update_news_sentiment()
