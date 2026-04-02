from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    DirectoryLoader,
    WebBaseLoader
)

import os
import tempfile


class DocumentLoader:

    def load_pdf(self, path: str) -> List[Document]:
        loader = PyPDFLoader(path)
        return loader.load()

    def load_txt(self, path: str) -> List[Document]:
        loader = TextLoader(path, encoding="utf-8")
        return loader.load()

    def load_directory(self, path: str, glob: str = "**/*") -> List[Document]:
        loader = DirectoryLoader(path, glob=glob, loader_cls=PyPDFLoader)
        return loader.load()

    def load_web(self, url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        return loader.load()

    def load_file(self, path: str) -> List[Document]:
        ext = Path(path).suffix.lower()

        if ext == ".pdf":
            return self.load_pdf(path)

        elif ext in [".txt", ".md"]:
            return self.load_txt(path)

        else:
            loader = UnstructuredFileLoader(path)
            return loader.load()

    def load_uploaded_files(self, uploaded_files):
        docs = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name

            loaded_docs = self.load_file(file_path)
            docs.extend(loaded_docs)

            os.remove(file_path)

        return docs