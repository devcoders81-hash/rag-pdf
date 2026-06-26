from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):
        self.model = CrossEncoder(
            "BAAI/bge-reranker-base"
        )

    def rerank(self, query, docs, top_k=5):

        pairs = [
            (query, d.page_content)
            for d in docs
        ]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [d for d, _ in ranked[:top_k]]