from src.ingestion.data_ingestion import (
    DataIngestionManager
)

from src.retrieval.vector_manager import (
    ChromaPDFIngestor
)

from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

# ------------------
# INGESTION
# ------------------

path = (
    "data/HR Policy Manual 2023.pdf"
)

data_ingestion = (
    DataIngestionManager()
)

vector_manager = (
    ChromaPDFIngestor()
)

docs = (
    data_ingestion.load_pdf(path)
)

chunks = (
    data_ingestion.chunk_strategy(
        docs
    )
)

vector_manager.ingest_pdf(
    path,
    chunks
)

# ------------------
# RETRIEVAL
# ------------------

retriever = HybridRetriever(
    vector_manager.vectorstore
)

results = retriever.search(
    "probation leave policy"
)

for i, doc in enumerate(
    results,
    start=1
):
    print(
        f"\nResult {i}"
    )

    print(
        doc.page_content[:500]
    )