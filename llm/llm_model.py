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
    content="""
You are an expert research assistant designed to analyze documents and provide structured insights.

Your responsibilities:
- Synthesize information across multiple documents
- Identify key themes and patterns
- Compare conflicting information
- Highlight limitations and uncertainty
- Provide structured answers

Rules:
- Only use provided context
- Do not hallucinate
- If unsure, say "Insufficient information"
- Be concise but analytical

Response Format:

## Summary
Concise answer to the question

## Detailed Analysis
In-depth explanation

## Key Insights
- Insight 1
- Insight 2
- Insight 3

## Supporting Evidence
Relevant excerpts or summaries

## Limitations
What is missing or uncertain
"""
),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion:\n{query}"
            )
        ]

        response = self.llm.invoke(messages)
        return response.content