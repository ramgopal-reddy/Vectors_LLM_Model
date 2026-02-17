import os
import re
import json
import numpy as np
import pandas as pd
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ==============================
# CONFIG
# ==============================

GENAI_API_KEY = os.getenv("GENAI_API_KEY")
FAISS_INDEX_PATH = "flipkart_index1.faiss"
DATA_PATH = "flipkart_data1.pkl"

genai.configure(api_key=GENAI_API_KEY)

# ==============================
# GLOBAL CACHE (Cache-AG Method)
# ==============================

app = FastAPI()

embedding_model = None
faiss_index = None
df = None
gemini_model = None


# ==============================
# STARTUP LOAD (Runs Once)
# ==============================

@app.on_event("startup")
def load_models():
    global embedding_model, faiss_index, df, gemini_model

    print("Loading embedding model...")
    embedding_model = SentenceTransformer(
        "paraphrase-MiniLM-L3-v2",
        device="cpu"
    )

    print("Loading FAISS index...")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    print("Loading dataset...")
    df = pd.read_pickle(DATA_PATH)

    print("Loading Gemini model...")
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")

    print("✅ All models loaded successfully")


# ==============================
# REQUEST MODEL
# ==============================

class QueryRequest(BaseModel):
    query: str


# ==============================
# GEMINI PARAM EXTRACTION
# ==============================

def extract_query_params(user_query: str):
    prompt = f"""
    Extract structured parameters from the user query.
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
# ==============================
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


# ==============================
# FILTERING
# ==============================

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


# ==============================
# CONFIDENCE SCORING
# ==============================

def compute_confidence_scores(candidates, distances, params):
    results = []

    for idx, row in candidates.iterrows():
        distance = distances.get(idx, 1)

        semantic_score = 1 / (1 + distance)

        try:
            rating_score = float(row["rating"]) / 5
        except:
            rating_score = 0.5

        price_score = 0.5
        if params.get("max_price") and row["price"]:
            if row["price"] <= params["max_price"]:
                price_score = 1 - (row["price"] / params["max_price"])
            else:
                price_score = 0

        brand_score = 0.5
        if params.get("brand"):
            brand_score = (
                1
                if params["brand"].lower() in row["supplier_name"].lower()
                else 0
            )

        final_score = (
            0.5 * semantic_score
            + 0.2 * rating_score
            + 0.2 * price_score
            + 0.1 * brand_score
        )

        results.append({
            "product_name": row["product_name"],
            "price": row["price"],
            "rating": row["rating"],
            "brand": row["supplier_name"],
            "confidence_score": round(float(final_score), 4)
        })

    results = sorted(results, key=lambda x: x["confidence_score"], reverse=True)

    return results[:5]


# ==============================
# SEARCH ENDPOINT
# ==============================

@app.post("/search")
def search_products(request: QueryRequest):

    params = extract_query_params(request.query)

    semantic_text = params.get("product") or request.query

    query_vector = embedding_model.encode([semantic_text])
    D, I = faiss_index.search(
        np.array(query_vector).astype("float32"), k=20
    )

    candidates = df.iloc[I[0]].copy()

    distances = dict(zip(candidates.index, D[0]))

    filtered = apply_filters(candidates, params)

    ranked_results = compute_confidence_scores(
        filtered, distances, params
    )

    return {
        "query": request.query,
        "parsed_params": params,
        "results": ranked_results
    }


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/")
def root():
    return {"status": "API running"}
