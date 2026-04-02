

from typing import List, Optional, Union
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

class VectorStore:
    """
    vectorstore_type: "chroma" (default) or "faiss"
    embeddings: a langchain-compatible embeddings object
    persist_directory: used for Chroma persistence; for FAISS this will save index files if save/load used
    """

    def __init__(
        self,
        embeddings,
        vectorstore_type: str = "chroma",
        persist_directory: str = "./chroma_db",
    ):
        self.embeddings = embeddings
        self.vectorstore_type = vectorstore_type.lower()
        self.persist_directory = persist_directory
        self.vectorstore = None

    def create_vectorstore(self, docs: List[Document], **kwargs):
        if self.vectorstore_type == "chroma":
            # Chroma persists to disk and manages its own embeddings param name in some versions
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                **kwargs,
            )
        elif self.vectorstore_type == "faiss":
            # FAISS builds an in-memory index; pass any kwargs you need
            self.vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings,
                allow_dangerous_deserialization=True,
                **kwargs,
            )
        else:
            raise ValueError("Unsupported vectorstore_type. Use 'chroma' or 'faiss'.")

        return self.vectorstore

    def load_vectorstore(self):
        if self.vectorstore_type== "chroma":
            # Chroma often loads via constructor pointing at the persist_directory
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
        elif self.vectorstore_type == "faiss":
            # FAISS can load from saved files
            self.vectorstore = FAISS.load_local(self.persist_directory, self.embeddings)
        else:
            raise ValueError("Unsupported vectorstore_type. Use 'chroma' or 'faiss'.")
        return self.vectorstore
    
    def save(self):
        if self.vectorstore_type == "faiss" and self.vectorstore:
            self.vectorstore.save_local(self.persist_directory)

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        if not self.vectorstore:
            raise ValueError("Vectorstore not created or loaded yet")
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs or {"k": 10})
