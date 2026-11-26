"""Tests for the modular RAG analytics assistant."""

from __future__ import annotations

from src.rag.assistant import ChurnRAGAssistant


def test_rag_assistant_retrieves_grounded_project_chunks():
    assistant = ChurnRAGAssistant()
    assert len(assistant.chunks) >= 4

    resp = assistant.answer("Which contract type has the highest churn rate?")
    assert resp["mode"] in {"local_grounded_rag", "llm_rag_groq", "llm_rag_openai"}
    assert "Retrieved Project Evidence" in resp["answer"]
    assert "Month-to-month" in resp["answer"]
    assert len(resp["retrieved_chunks"]) >= 1
