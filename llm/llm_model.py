from typing import List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage


class LLMModel:
    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
    ):
        self.llm = ChatGroq(
            model=model_name,
            temperature=temperature
        )

    def generate(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in docs])

        messages = [
            SystemMessage(
                content="Answer the question using the provided context only."
            ),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion:\n{query}"
            )
        ]

        response = self.llm.invoke(messages)
        return response.content