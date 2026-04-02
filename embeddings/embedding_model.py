from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Optional


class EmbeddingModel:
    """
    provider: "openai", "huggingface"
    model_name: provider-specific model identifier
    hf_model_kwargs: optional dict passed to HuggingFaceEmbeddings
    """

    def __init__(
        self,
        provider: str = "hf",
        model_name: Optional[str] = None,
        hf_model_kwargs: Optional[dict] = None,
    ):
        self.provider = provider.lower()

        if self.provider == "openai":
            model = model_name or "text-embedding-3-small"
            self.embeddings = OpenAIEmbeddings(model=model)

        elif self.provider in ("huggingface", "hf"):
            model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs=hf_model_kwargs or {}
            )

        else:
            raise ValueError(
                "Unsupported provider. Use 'openai' or 'huggingface'."
            )

    def get_embeddings(self):
        return self.embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        if hasattr(self.embeddings, "embed_query"):
            return self.embeddings.embed_query(text)

        return self.embeddings.embed_documents([text])[0]