from typing import List
from langchain_classic.retrievers import EnsembleRetriever

class MergerRetriever:
    def merge(self, retrievers: List, weights=None):
        if weights is None:
            weights = [1/len(retrievers)] * len(retrievers)
        return EnsembleRetriever(retrievers=retrievers, weights=weights)