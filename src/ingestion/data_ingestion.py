from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
from pathlib import Path


class DataIngestionManager:

    def __init__(self):
        pass

    # -----------------------------
    # LOAD PDF
    # -----------------------------
    def load_pdf(self, path):
        loader = PyPDFLoader(file_path=path)
        docs = loader.load()
        return docs

    # -----------------------------
    # CHUNKING + METADATA FIX
    # -----------------------------
    def chunk_strategy(self, docs, file_path=None):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

        all_splits = text_splitter.split_documents(docs)

        file_name = Path(file_path).stem if file_path else "unknown_file"

        for i, chunk in enumerate(all_splits):

            # ✅ UNIQUE CHUNK ID (CRITICAL FIX)
            chunk_id = hashlib.sha256(
                f"{file_name}_{i}_{chunk.page_content}".encode()
            ).hexdigest()

            chunk.metadata["id"] = chunk_id

            # ✅ SOURCE TRACKING
            chunk.metadata["source_file"] = file_name
            chunk.metadata["chunk_index"] = i

        print(f"Split '{file_name}' into {len(all_splits)} sub-documents.")

        if len(all_splits) > 0:
            print(f"Sample chunk: {all_splits[0]}")

        return all_splits