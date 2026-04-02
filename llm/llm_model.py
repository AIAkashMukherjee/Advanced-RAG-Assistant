from typing import List
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage


class LLMModel:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
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
You are a research assistant specializing in machine learning and AI.

Your responses must:
- Be technical and analytical
- Compare architectures when relevant
- Explain theoretical tradeoffs
- Discuss limitations explicitly
- Use structured sections
- Avoid generic summaries

Response Format:

## Summary

## Technical Explanation

## Comparison (if applicable)

## Advantages

## Limitations

## Practical Implications

## Conclusion
"""
),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion:\n{query}"
            )
        ]

        response = self.llm.invoke(messages)
        return response.content