from fastapi import FastAPI, UploadFile, File
import shutil
import os

from src.ingestion.data_ingestion import DataIngestionManager
from src.retrieval.vector_manager import VectorStoreManager
from src.retrieval.hybrid_retriever_rrf import RRFHybridRetriever
from src.retrieval.reranker import Reranker
from src.llm.generator import generate_answer


app = FastAPI()

ingestion = DataIngestionManager()
vector = VectorStoreManager()
retriever = RRFHybridRetriever(vector.db)
reranker = Reranker()


# -------------------------
# INGEST
# -------------------------
@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):

    os.makedirs("temp", exist_ok=True)

    path = f"temp/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = ingestion.load_pdf(path)
    chunks = ingestion.chunk_strategy(docs)

    vector.ingest(path, chunks)

    retriever.build_index()

    return {
        "message": "ingested",
        "chunks": len(chunks)
    }


# -------------------------
# QUERY
# -------------------------
@app.post("/query")
def query(payload: dict):

    query = payload["query"]

    # 1. Hybrid RRF retrieval
    docs = retriever.search(query, k=10)

    # 2. Reranking (IMPORTANT)
    docs = reranker.rerank(query, docs, top_k=5)

    # 3. LLM answer
    answer = generate_answer(query, docs)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "text": d.page_content[:200],
                "source": d.metadata.get("source")
            }
            for d in docs
        ]
    }