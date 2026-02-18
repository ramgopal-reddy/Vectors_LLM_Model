import os
import json
import re
import hashlib
import requests
import numpy as np
import pandas as pd
import faiss
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai


# =====================================================
# CONFIG
# =====================================================

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
CLOUDFLARE_URL = os.getenv("CLOUDFLARE_URL")

FAISS_INDEX_PATH = "bge_index.faiss"
DATA_PATH = "bge_data.pkl"

TOP_K = 20

if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY not set")

if not CLOUDFLARE_URL:
    raise RuntimeError("CLOUDFLARE_URL not set")

genai.configure(api_key=GENAI_API_KEY)


# =====================================================
# APP INIT (Cache-AG Global Cache)
# =====================================================

app = FastAPI()

faiss_index = None
df = None
gemini_model = None


# =====================================================
# STARTUP LOAD (Cache-AG Core)
# =====================================================

@app.on_event("startup")
def load_data():
    global faiss_index, df, gemini_model

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    print("Loading dataset...")
    df = pd.read_pickle(DATA_PATH)

    # Ensure numeric types once (not per request)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    print("Loading Gemini model...")
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    print("✅ Startup complete")


# =====================================================
# REQUEST MODEL
# =====================================================

class QueryRequest(BaseModel):
    query: str


# =====================================================
# LRU QUERY CACHE (Cache-AG Layer 1)
# =====================================================

@lru_cache(maxsize=256)
def cached_search(query_text: str):
    return perform_search(query_text)


# =====================================================
# GEMINI PARAM EXTRACTION
# =====================================================

def extract_query_params(user_query: str):

    prompt = f"""
    Extract structured shopping parameters.
    Return ONLY valid JSON.

    Query: "{user_query}"

    JSON format:
    {{
      "product": string or null,
      "max_price": number or null,
      "min_rating": number or null,
      "category": string or null,
      "brand": string or null
    }}
    """


    response = gemini_model.generate_content(prompt)
    text = response.text.strip()

    text = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except:
        return {
            "product": user_query,
            "max_price": None,
            "min_rating": None,
            "category": None,
            "brand": None
        }


# =====================================================
# EMBEDDING CACHE (Cache-AG Layer 2)
# =====================================================

embedding_cache = {}

def get_embedding(text: str):

    key = hashlib.md5(text.encode()).hexdigest()

    if key in embedding_cache:
        return embedding_cache[key]

    try:
        response = requests.post(
            CLOUDFLARE_URL,
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()

        vector = np.array(response.json()["data"][0]).astype("float32")

        # Normalize for cosine similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        embedding_cache[key] = vector

        # prevent unlimited growth
        if len(embedding_cache) > 1000:
            embedding_cache.clear()

        return vector

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


# =====================================================
# FILTERING
# =====================================================

def apply_filters(dataframe, params):

    filtered = dataframe

    # 1️⃣ Numeric Filters (Safe but strict)
    if params.get("max_price") is not None:
        temp = filtered[filtered["price"] <= float(params["max_price"])]
        if not temp.empty:
            filtered = temp

    if params.get("min_rating") is not None:
        temp = filtered[filtered["rating"] >= float(params["min_rating"])]
        if not temp.empty:
            filtered = temp

    # 2️⃣ Category Filter (Soft)
    if params.get("category"):
        temp = filtered[
            filtered["clean_category"].astype(str).str.contains(
                str(params["category"]), case=False, na=False
            )
        ]
        if not temp.empty:
            filtered = temp

    # 3️⃣ Brand Filter (Soft)
    if params.get("brand"):
        temp = filtered[
            filtered["supplier_name"].astype(str).str.contains(
                str(params["brand"]), case=False, na=False
            )
        ]
        if not temp.empty:
            filtered = temp

    return filtered



# =====================================================
# CORE SEARCH LOGIC
# =====================================================

def perform_search(user_query: str):

    params = extract_query_params(user_query)
    semantic_text = params.get("product") or user_query

    query_vector = get_embedding(semantic_text).reshape(1, -1)

    # Cosine similarity search (IndexFlatIP)
    scores, indices = faiss_index.search(query_vector, k=TOP_K)

    candidates = df.iloc[indices[0]].copy()
    candidates["similarity"] = scores[0]


    fallback_used = False
    
    filtered = apply_filters(candidates, params)

# 🔥 Fallback: if everything filtered out, use semantic-only
    if filtered.empty:
        filtered = candidates.copy()
        fallback_used = True

    filtered = filtered.sort_values("similarity", ascending=False)


    results = []

    for _, row in filtered.head(TOP_K).iterrows():

    price = row.get("price")
    rating = row.get("rating")

    # Fix NaN for JSON
    if pd.isna(price):
        price = None
    else:
        price = float(price)

    if pd.isna(rating):
        rating = None
    else:
        rating = float(rating)

    results.append({
        "product_name": row.get("product_name"),
        "price": price,
        "rating": rating,
        "brand": row.get("supplier_name"),
        "similarity_score": round(float(row["similarity"]), 4)
    })


    return {
        "query": user_query,
        "parsed_params": params,
        "fallback_used": fallback_used,
        "results": results
    }



# =====================================================
# SEARCH ENDPOINT
# =====================================================

@app.post("/search")
def search_products(request: QueryRequest):
    return cached_search(request.query)


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def health():
    return {
        "status": "API running",
        "vectors": faiss_index.ntotal
    }
