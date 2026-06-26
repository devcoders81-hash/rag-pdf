import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStoreManager:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.db = Chroma(
            collection_name="rag_docs",
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    # ---------------------------
    # FILE HASH
    # ---------------------------
    def _file_hash(self, path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    # ---------------------------
    # CHECK IF FILE EXISTS
    # ---------------------------
    def file_exists(self, file_hash: str) -> bool:

        result = self.db.get(
            where={"file_hash": file_hash}
        )

        return len(result["ids"]) > 0

    # ---------------------------
    # INGEST DATA
    # ---------------------------
    def ingest(self, path: str, chunks):

        if not chunks:
            print("No chunks to ingest.")
            return

        file_hash = self._file_hash(path)

        # ✅ SKIP IF EXISTS
        if self.file_exists(file_hash):
            print("Data already exists in Chroma DB. Skipping ingestion.")
            return

        ids = []

        for chunk in chunks:

            chunk.metadata["file_hash"] = file_hash
            chunk.metadata["source"] = Path(path).name

            # deterministic chunk id
            chunk_id = hashlib.sha256(
                chunk.page_content.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

        self.db.add_documents(
            documents=chunks,
            ids=ids
        )

        print(f"Ingested {len(chunks)} chunks successfully.")

    # ---------------------------
    # SEMANTIC SEARCH
    # ---------------------------
    def search(self, query: str, k: int = 5):

        return self.db.similarity_search(
            query,
            k=k
        )