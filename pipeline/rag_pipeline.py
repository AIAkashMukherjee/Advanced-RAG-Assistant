from loader.document_loader import DocumentLoader
from chunking.text_splitter import TextChunker
from embeddings.embedding_model import EmbeddingModel
from vectorstore.vectorstoredb import VectorStore
from retrievers.hybrid_retriever import HybridRetriever
from retrievers.merge_retriever import MergerRetriever
from retrievers.parent_retriever import ParentRetriever
from reranker.reranker import Reranker
from reorder.long_context_reorder import LongContextReorderWrapper
from llm.llm_model import LLMModel

class AdvancedRAGPipeline:
    def __init__(self, config):
        self.config=config
        self.loader = DocumentLoader()
        self.chunker = TextChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.embedding_model = EmbeddingModel(provider="huggingface", model_name=config.EMBEDDING_MODEL)
        self.vectorstore = VectorStore(self.embedding_model.get_embeddings())
        self.reranker=Reranker()
        self.long_context_reorder=LongContextReorderWrapper()
        self.llm=LLMModel(model_name=config.LLM_MODEL)
        self.retriever = None

    def ingest(self, source_path: str):
        docs = self.loader.load_directory(source_path)
        chunks = self.chunker.split_documents(docs)
        print("INGEST RUNNING")
        # create vectorstore
        vectorstore = self.vectorstore.create_vectorstore(chunks)

        # 1. vector retriever
        vector_retriever = vectorstore.as_retriever()

        # 2. hybrid retriever
        hybrid = HybridRetriever(vector_retriever, chunks).get_retriever()

        # 3. parent retriever
        parent = ParentRetriever(vectorstore)
        parent_retriever = parent.add_documents(chunks)

        # 4. merge retriever
        merger = MergerRetriever()
        self.retriever = merger.merge(
            [hybrid, parent_retriever],
            weights=[0.5, 0.5]
        )
        print(f"✅ Ingestion complete: {len(chunks)} chunks from {source_path}")

    def ingest_uploaded_files(self, uploaded_files):
        docs = self.loader.load_uploaded_files(uploaded_files)
        chunks = self.chunker.split_documents(docs)
        vectorstore = self.vectorstore.create_vectorstore(chunks)

        # 1. vector retriever
        vector_retriever = vectorstore.as_retriever()

        # 2. hybrid retriever
        hybrid = HybridRetriever(vector_retriever, chunks).get_retriever()

        # 3. parent retriever
        parent = ParentRetriever(vectorstore)
        parent_retriever = parent.add_documents(chunks)

        # 4. merge retriever
        merger = MergerRetriever()
        self.retriever = merger.merge(
            [hybrid, parent_retriever],
            weights=[0.5, 0.5]
        )

        print(f"✅ Uploaded files ingested: {len(chunks)} chunks")
    def run(self, query: str):
        if self.retriever is None:
            raise ValueError("Run ingest() or ingest_uploaded_files() first before querying.")
        
        # 1. Retrieve
        docs = self.retriever.invoke(query)

        # 2. Rerank
        compression_retriever = self.reranker.get_compression_retriever(self.retriever)
        reranked = compression_retriever.invoke(query)

        # 3. Reorder for long context
        reordered = self.long_context_reorder.reorder(reranked)

        # 4. Generate
        context = "\n\n".join([doc.page_content for doc in reordered])
        answer = self.llm.generate(query, context)

        return {"answer": answer, "sources": reordered}
