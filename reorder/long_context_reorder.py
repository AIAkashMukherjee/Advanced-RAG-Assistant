from langchain_community.document_transformers import LongContextReorder


class LongContextReorderWrapper:
    def reorder(self, docs):
        reordering = LongContextReorder()
        return reordering.transform_documents(docs)