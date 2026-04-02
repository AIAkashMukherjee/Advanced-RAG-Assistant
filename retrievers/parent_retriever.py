
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ParentRetriever:
    def __init__(self, vectorstore, child_chunk_size=400, parent_chunk_size=2000):
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)
        self.store = InMemoryStore()
        self.retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=self.store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

    def add_documents(self, docs):    
        self.retriever.add_documents(docs)
        return self.retriever
    