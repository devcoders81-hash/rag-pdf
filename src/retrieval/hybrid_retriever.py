from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class HybridRetriever:

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.documents = []
        self.bm25 = None
        self.build_index()

    def build_index(self):

        data = self.vectorstore.get()

        self.documents = [
            Document(
                page_content=text,
                metadata=meta
            )
            for text, meta in zip(
                data["documents"],
                data["metadatas"]
            )
        ]

        tokenized = [
            d.page_content.lower().split()
            for d in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized)

        print(f"BM25 built on {len(self.documents)} docs.")

    def search(self, query, k=5):

        semantic = self.vectorstore.similarity_search(
            query,
            k=k
        )

        keyword = self.bm25.get_top_n(
            query.lower().split(),
            self.documents,
            n=k
        )

        seen = set()
        results = []

        for doc in semantic + keyword:

            if doc.page_content not in seen:
                seen.add(doc.page_content)
                results.append(doc)

        return results