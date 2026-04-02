from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import List

class HybridRetriever:
    def __init__(self, vector_retriever, documents: List[Document], weights=None):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.weights = weights or [0.6, 0.4]

    def get_retriever(self):
        return EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=self.weights
        )    