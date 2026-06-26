import streamlit as st
import pandas as pd

from src.retrieval.vector_manager import VectorStoreManager
from src.retrieval.hybrid_retriever_rrf import RRFHybridRetriever
from src.retrieval.reranker import Reranker

st.title("📊 RAG Evaluation Dashboard")

# -----------------------------
# INIT RAG COMPONENTS
# -----------------------------
vector = VectorStoreManager()
retriever = RRFHybridRetriever(vector.db)
reranker = Reranker()

# -----------------------------
# TEST SET (IMPORTANT CHANGE)
# You MUST use real chunk IDs here
# -----------------------------
test_cases = [
    {
        "query": "probation leave policy",
        "relevant_ids": ["chunk_1", "chunk_5"]
    },
    {
        "query": "working hours policy",
        "relevant_ids": ["chunk_2", "chunk_8"]
    },
    {
        "query": "dress code policy",
        "relevant_ids": ["chunk_3", "chunk_7"]
    }
]

# -----------------------------
# REAL METRICS (FIXED)
# -----------------------------
def precision_at_k(retrieved_ids, relevant_ids):
    if not retrieved_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(relevant_ids))
    return hits / len(retrieved_ids)


def recall_at_k(retrieved_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids) & set(relevant_ids))
    return hits / len(relevant_ids)

# -----------------------------
# REAL RAG RETRIEVAL FUNCTION
# -----------------------------
def rag_retrieve(query):

    # 1. Hybrid retrieval
    docs = retriever.search(query, k=10)

    # 2. Rerank
    docs = reranker.rerank(query, docs, top_k=5)

    # 3. Return DOCUMENT IDS (IMPORTANT)
    retrieved_ids = [
        d.metadata.get("id", None) for d in docs
    ]

    return retrieved_ids

# -----------------------------
# RUN EVALUATION
# -----------------------------
results = []

for test in test_cases:

    query = test["query"]
    relevant_ids = test["relevant_ids"]

    retrieved_ids = rag_retrieve(query)

    precision = precision_at_k(retrieved_ids, relevant_ids)
    recall = recall_at_k(retrieved_ids, relevant_ids)

    results.append({
        "query": query,
        "precision@k": round(precision, 3),
        "recall@k": round(recall, 3)
    })

df = pd.DataFrame(results)

# -----------------------------
# DASHBOARD UI
# -----------------------------
st.dataframe(df)

st.metric(
    "Avg Precision@K",
    round(df["precision@k"].mean(), 2)
)

st.metric(
    "Avg Recall@K",
    round(df["recall@k"].mean(), 2)
)