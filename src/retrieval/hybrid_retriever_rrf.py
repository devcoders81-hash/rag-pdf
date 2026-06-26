from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class RRFHybridRetriever:

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.docs = []
        self.bm25 = None
        #self.build_index()

    # def build_index(self):

    #     data = self.vectorstore.get()

    #     self.docs = data["documents"]
    #     self.metadatas = data["metadatas"]

    #     tokenized = [
    #         doc.lower().split()
    #         for doc in self.docs
    #     ]

    #     self.bm25 = BM25Okapi(tokenized)

    #     print(f"Index built: {len(self.docs)} docs")
    
    def build_index(self):

        data = self.vectorstore.get()

        if not data["documents"]:
            print("⚠️ No documents in Chroma yet. Skipping BM25 build.")
            self.bm25 = None
            self.docs = []
            return

        self.docs = data["documents"]

        tokenized = [
            doc.lower().split()
            for doc in self.docs
        ]

        self.bm25 = BM25Okapi(tokenized)

        print(f"BM25 built on {len(self.docs)} docs")

    def _rrf(self, sem, key, k=60):

        scores = {}

        def add(results,weight=1.0):
            for r, doc in enumerate(results):
                key=doc.page_content
                scores[key] = scores.get(key, 0) + weight * (1 / (k + r))

        add(sem,weight=0.7)
        add(key,weight=0.3)

        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [Document(page_content=d[0]) for d in sorted_docs]

    # def search(self, query, k=10):

    #     semantic = self.vectorstore.similarity_search(
    #         query, k=k
    #     )

    #     keyword = self.bm25.get_top_n(
    #         query.lower().split(),
    #         self.vectorstore.get()["documents"],
    #         n=k
    #     )

    #     return self._rrf(semantic, keyword)
    # def search(self, query, k=10):

    #     if self.bm25 is None:
    #         self.build_index()

    #     semantic = self.vectorstore.similarity_search(
    #         query, k=k
    #     )

    #     keyword_docs = []

    #     if self.bm25:
    #         bm25_results = self.bm25.get_top_n(
    #             query.lower().split(),
    #             self.docs,
    #             n=k
    #         )

    #         # convert strings → Document objects
    #         keyword_docs = [
    #             Document(page_content=text)
    #             for text in bm25_results
    #         ]

    #     # return ONLY Document objects
    #     return semantic + keyword_docs
    
    def search(self, query, k=10):

        if self.bm25 is None:
            self.build_index()

        semantic = self.vectorstore.similarity_search(query, k=k)

        bm25_results = self.bm25.get_top_n(
            query.lower().split(),
            self.docs,
            n=k
        )

        bm25_docs = [
            Document(page_content=t)
            for t in bm25_results
        ]

        return self._rrf(semantic, bm25_docs)