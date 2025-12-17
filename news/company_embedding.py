# #!/usr/bin/env python3
# """
# build_company_embeddings.py
# ---------------------------
# Builds vector embeddings for each company's description using OpenAI's
# text-embedding-3-small model.

# ✅ Input: company_profiles.json
# ✅ Output: company_embeddings.pkl (for fast semantic matching later)
# """

# import os
# import json
# import pickle
# from tqdm import tqdm
# from openai import OpenAI
# from dotenv import load_dotenv

# # === Load environment ===
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     raise ValueError("⚠️ OPENAI_API_KEY missing in .env")
# client = OpenAI(api_key=OPENAI_API_KEY)

# # === Paths ===
# PROFILE_PATH = "./data/company_profiles.json"
# OUTPUT_PATH = "company_embeddings.pkl"

# MODEL = "text-embedding-3-small"

# # === Load company profiles ===
# if not os.path.exists(PROFILE_PATH):
#     raise FileNotFoundError("❌ Missing company_profiles.json — build it first!")

# with open(PROFILE_PATH, "r", encoding="utf-8") as f:
#     profiles = json.load(f)

# print(f"[📘] Loaded {len(profiles)} company profiles")

# embeddings = {}

# # === Build embeddings ===
# for symbol, data in tqdm(profiles.items(), desc="Building embeddings"):
#     desc = data.get("description") or ""
#     if not desc.strip():
#         continue

#     try:
#         response = client.embeddings.create(
#             model=MODEL, input=desc[:2000]  # truncate long text
#         )
#         vec = response.data[0].embedding
#         embeddings[symbol] = vec
#     except Exception as e:
#         print(f"[⚠️] Failed to embed {symbol}: {e}")

# # === Save embeddings ===
# os.makedirs("news", exist_ok=True)
# with open(OUTPUT_PATH, "wb") as f:
#     pickle.dump(embeddings, f)

# print(f"[💾] Saved {len(embeddings)} embeddings → {OUTPUT_PATH}")


#!/usr/bin/env python3
"""
build_company_embeddings.py
-----------------------------------------------
Builds company symbol embeddings using OpenAI's
"text-embedding-3-large" model for semantic similarity mapping.

✅ Input:
    - company_profiles.json (symbol → description)
✅ Output:
    - company_embeddings.pkl (aligned 3072-dim vectors)

Usage:
    python build_company_embeddings.py
"""

import os
import json
import pickle
from tqdm import tqdm
from dotenv import load_dotenv

# === Modern imports (LangChain v0.2+ compliant) ===
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    from langchain.embeddings import OpenAIEmbeddings  # fallback for older setups

# === Load environment ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ OPENAI_API_KEY missing in .env")

# === Paths ===
PROFILE_PATH = "./data/company_profiles.json"
OUTPUT_PATH = "./data/company_embeddings.pkl"
MODEL = "text-embedding-3-large"  # must match your main pipeline model (3072-dim)

# === Load company profiles ===
if not os.path.exists(PROFILE_PATH):
    raise FileNotFoundError(
        "❌ Missing ./data/company_profiles.json — create it first!"
    )

with open(PROFILE_PATH, "r", encoding="utf-8") as f:
    profiles = json.load(f)

print(f"[📘] Loaded {len(profiles)} company profiles")

# === Initialize embedding function ===
embedding_fn = OpenAIEmbeddings(model=MODEL, openai_api_key=OPENAI_API_KEY)

embeddings = {}

# === Build embeddings ===
for symbol, data in tqdm(profiles.items(), desc="Building embeddings"):
    desc = (data.get("description") or symbol).strip()
    if not desc:
        continue

    try:
        vec = embedding_fn.embed_query(desc[:2000])  # safe truncation
        embeddings[symbol] = vec
    except Exception as e:
        print(f"[⚠️] Failed to embed {symbol}: {e}")

# === Save embeddings ===
os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(embeddings, f)

print(f"[💾] Saved {len(embeddings)} embeddings → {OUTPUT_PATH}")
print(f"[✅] Model: {MODEL} | Dimension: {len(next(iter(embeddings.values())))}")
