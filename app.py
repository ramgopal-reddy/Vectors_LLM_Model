import os
import json
import re
import requests
import numpy as np
import pandas as pd
import faiss

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai


# =====================================================
# CONFIG
# =====================================================

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
CLOUDFLARE_URL = os.getenv("CLOUDFLARE_URL")

FAISS_INDEX_PATH = "flipkart_index1.faiss"
DATA_PATH = "flipkart_data1.pkl"

TOP_K = 5

if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY not set")

if not CLOUDFLARE_URL:
    raise RuntimeError("CLOUDFLARE_URL not set")

genai.configure(api_key=GENAI_API_KEY)


# =====================================================
# APP INIT
# =====================================================

app = FastAPI()

faiss_index = None
df = None
gemini_model = None


# =====================================================
# LOAD LIGHTWEIGHT COMPONENTS ON STARTUP
# =====================================================

@app.on_event("startup")
def load_data():
    global faiss_index, df, gemini_model

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    print("Loading dataset...")
    df = pd.read_pickle(DATA_PATH)

    print("Loading Gemini model...")
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    print("✅ Startup complete")


# =====================================================
# REQUEST MODEL
# =====================================================

class QueryRequest(BaseModel):
    query: str


# =====================================================
# GEMINI PARAM EXTRACTION
# =====================================================

def extract_query_params(user_query: str):

    prompt = f"""
    Extract structured shopping parameters from the query.
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

# =====================================================

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
# CLOUDFLARE EMBEDDING CALL
# =====================================================

def get_embedding(text: str):

    try:
        response = requests.post(
            CLOUDFLARE_URL,
            json={"text": text},
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        vector = data["data"][0]

        return np.array(vector).astype("float32")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")


# =====================================================
# FILTERING
# =====================================================

def apply_filters(dataframe, params):

    filtered = dataframe.copy()

    if params.get("max_price"):
        filtered = filtered[filtered["price"] <= params["max_price"]]

    if params.get("min_rating"):
        filtered = filtered[filtered["rating"] >= params["min_rating"]]

    if params.get("category"):
        filtered = filtered[
            filtered["clean_category"].str.contains(
                params["category"], case=False, na=False
            )
        ]

    if params.get("brand"):
        filtered = filtered[
            filtered["supplier_name"].str.contains(
                params["brand"], case=False, na=False
            )
        ]

    return filtered


# =====================================================
# CONFIDENCE SCORING
# =====================================================

def rank_results(candidates, distances, params):

    results = []

    for idx, row in candidates.iterrows():

        distance = distances.get(idx, 1)
        semantic_score = 1 / (1 + distance)

        rating_score = float(row.get("rating", 0)) / 5 if row.get("rating") else 0.5

        price_score = 0.5
        if params.get("max_price") and row.get("price"):
            if row["price"] <= params["max_price"]:
                price_score = 1 - (row["price"] / params["max_price"])
            else:
                price_score = 0

        final_score = (
            0.6 * semantic_score +
            0.2 * rating_score +
            0.2 * price_score
        )

        results.append({
            "product_name": row.get("product_name"),
            "price": row.get("price"),
            "rating": row.get("rating"),
            "brand": row.get("supplier_name"),
            "confidence_score": round(float(final_score), 4)
        })

    results = sorted(results, key=lambda x: x["confidence_score"], reverse=True)

    return results[:TOP_K]


# =====================================================
# SEARCH ENDPOINT
# =====================================================

@app.post("/search")
def search_products(request: QueryRequest):

    params = extract_query_params(request.query)

    semantic_text = params.get("product") or request.query

    query_vector = get_embedding(semantic_text).reshape(1, -1)

    D, I = faiss_index.search(query_vector, k=20)

    candidates = df.iloc[I[0]].copy()

    distances = dict(zip(candidates.index, D[0]))

    filtered = apply_filters(candidates, params)

    ranked = rank_results(filtered, distances, params)

    return {
        "query": request.query,
        "parsed_params": params,
        "results": ranked
    }


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def health():
    return {"status": "API running"}
