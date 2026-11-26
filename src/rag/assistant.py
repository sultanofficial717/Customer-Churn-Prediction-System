"""Modular RAG assistant grounded in real EDA, model evaluation, and SHAP artifacts.

Supports two operating modes:
1. Optional LLM + Vector RAG (ChromaDB + OpenAI/Groq via LangChain when optional
   dependencies in requirements-rag.txt and API credentials are configured).
2. Zero-API-Key Local Grounded Retrieval (TF-IDF cosine similarity over actual
   project artifacts) so the core application remains 100% functional offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    EDA_SUMMARY_PATH,
    EVALUATION_SUMMARY_PATH,
    METADATA_PATH,
    SHAP_SUMMARY_PATH,
)


@dataclass
class KnowledgeChunk:
    chunk_id: str
    source: str
    title: str
    content: str


def build_knowledge_chunks() -> List[KnowledgeChunk]:
    """Assemble factual text chunks directly from generated project artifacts."""
    chunks: List[KnowledgeChunk] = []

    if EDA_SUMMARY_PATH.exists():
        with open(EDA_SUMMARY_PATH, "r", encoding="utf-8") as f:
            eda = json.load(f)

        td = eda["target_distribution"]
        dq = eda["data_quality"]
        bm = eda["measured_billing_metrics"]
        chunks.append(
            KnowledgeChunk(
                chunk_id="eda_overview",
                source="reports/evaluation/eda_summary.json",
                title="Dataset Overview & Overall Churn Rate",
                content=(
                    f"The IBM Telco Customer Churn dataset contains {td['total_customers']:,} unique customers "
                    f"across {dq['raw_columns']} columns. Data quality check found {dq['duplicate_rows']} duplicate rows "
                    f"and {dq['blank_total_charges_count']} blank TotalCharges entries (all with tenure = 0, deterministically set to 0.0). "
                    f"Overall churn rate is {td['churn_rate']:.2%} ({td['churned_customers']:,} churned vs. "
                    f"{td['retained_customers']:,} retained; imbalance ratio {td['imbalance_ratio_no_to_yes']}:1). "
                    f"Measured total monthly charges across all accounts are ${bm['total_monthly_charges_sum']:,.2f}, "
                    f"with churned accounts representing ${bm['churned_monthly_charges_sum']:,.2f} "
                    f"({bm['churned_monthly_charges_share']:.1%} of total monthly billing). "
                    f"Average MonthlyCharges is ${bm['avg_monthly_charge_churned']:.2f} for churned customers vs. "
                    f"${bm['avg_monthly_charge_retained']:.2f} for retained customers."
                ),
            )
        )

        for seg_name, seg_rows in eda.get("segment_analysis", {}).items():
            lines = [
                f"- {row['segment']}: {row['churn_rate']:.1%} churn "
                f"({row['churned']:,}/{row['customers']:,} customers, avg monthly ${row['avg_monthly_charges']:.2f}, "
                f"avg tenure {row['avg_tenure']:.1f}m)"
                for row in seg_rows
            ]
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"eda_segment_{seg_name.lower()}",
                    source="reports/evaluation/eda_summary.json",
                    title=f"Churn Breakdown by {seg_name}",
                    content=f"Measured churn rates segmented by {seg_name}:\n" + "\n".join(lines),
                )
            )

    if EVALUATION_SUMMARY_PATH.exists():
        with open(EVALUATION_SUMMARY_PATH, "r", encoding="utf-8") as f:
            ev = json.load(f)

        tm = ev["selected_model_test_metrics"]
        cm = tm["confusion_matrix"]
        chunks.append(
            KnowledgeChunk(
                chunk_id="model_selection",
                source="reports/evaluation/evaluation_summary.json",
                title="Selected Model & Held-Out Test Evaluation",
                content=(
                    f"Selected production model: {ev['selected_model']} (version {ev['model_version']}). "
                    f"Selection criterion: {ev['selection_criterion']}. "
                    f"Evaluated on stratified 80/20 split ({ev['dataset_split']['train_rows']:,} train / "
                    f"{ev['dataset_split']['test_rows']:,} test). "
                    f"Held-out test metrics at 0.50 threshold: ROC-AUC = {tm['roc_auc']:.4f}, "
                    f"PR-AUC = {tm['pr_auc']:.4f}, F1 = {tm['f1_score']:.4f}, Recall = {tm['recall']:.4f}, "
                    f"Precision = {tm['precision']:.4f}, Accuracy = {tm['accuracy']:.4f}. "
                    f"Confusion matrix: TP={cm['tp']}, FN={cm['fn']}, FP={cm['fp']}, TN={cm['tn']}."
                ),
            )
        )

        model_lines = [
            f"- {m['model_name']}: CV ROC-AUC={m['cv_roc_auc']:.4f}, Test ROC-AUC={m['roc_auc']:.4f}, "
            f"PR-AUC={m['pr_auc']:.4f}, F1={m['f1_score']:.4f}, Recall={m['recall']:.4f}, Precision={m['precision']:.4f}"
            for m in ev.get("all_models", [])
        ]
        chunks.append(
            KnowledgeChunk(
                chunk_id="model_comparison",
                source="reports/evaluation/evaluation_summary.json",
                title="All Evaluated Models Comparison",
                content="Model comparison summary (sorted by CV performance):\n" + "\n".join(model_lines),
            )
        )

    if SHAP_SUMMARY_PATH.exists():
        with open(SHAP_SUMMARY_PATH, "r", encoding="utf-8") as f:
            sh = json.load(f)

        feat_lines = [
            f"- {item['display_name']}: mean |SHAP| = {item['mean_abs_shap']:.4f} "
            f"(mean signed SHAP = {item['mean_signed_shap']:+.4f})"
            for item in sh.get("top_global_features", [])[:12]
        ]
        chunks.append(
            KnowledgeChunk(
                chunk_id="shap_global",
                source="reports/evaluation/shap_summary.json",
                title="Global SHAP Feature Importance",
                content=(
                    f"Computed with {sh['explainer_type']} on {sh['evaluated_rows']:,} held-out test customers.\n"
                    "Top global churn predictors by mean absolute SHAP impact:\n"
                    + "\n".join(feat_lines)
                    + "\nNote: SHAP reflects model associations, not proven causal effects."
                ),
            )
        )

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        rt = meta.get("risk_thresholds", {})
        chunks.append(
            KnowledgeChunk(
                chunk_id="operational_thresholds",
                source="models/model_metadata.json",
                title="Operational Risk Thresholds & Retention Implications",
                content=(
                    f"Default operational risk tiers: Low (< {rt.get('low_max', 0.30):.0%}), "
                    f"Medium ({rt.get('low_max', 0.30):.0%}–{rt.get('high_min', 0.65):.0%}), "
                    f"High (>= {rt.get('high_min', 0.65):.0%}). {rt.get('note', '')} "
                    "Business trade-off: False negatives miss churning customers (lost recurring revenue), "
                    "whereas false positives spend retention outreach/discounts on customers who would have stayed."
                ),
            )
        )

    return chunks


class ChurnRAGAssistant:
    """Evidence-grounded assistant with optional ChromaDB + LLM (Groq/OpenAI) support."""

    def __init__(self, chunks: Optional[List[KnowledgeChunk]] = None):
        self.chunks = chunks if chunks is not None else build_knowledge_chunks()
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None
        if self.chunks:
            self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            corpus = [f"{c.title}\n{c.content}" for c in self.chunks]
            self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k factual chunks using local TF-IDF cosine similarity."""
        if not self.chunks or self._vectorizer is None or self._tfidf_matrix is None:
            return []

        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._tfidf_matrix).ravel()
        top_indices = sims.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "source": chunk.source,
                    "score": round(float(sims[idx]), 4),
                    "content": chunk.content,
                }
            )
        return results

    def _try_llm_synthesis(
        self,
        question: str,
        context_text: str,
        customer_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt LLM synthesis via Groq or OpenAI if credentials and packages exist."""
        provider = os.getenv("LLM_PROVIDER", "groq").lower()
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        openai_key = os.getenv("OPENAI_API_KEY", "").strip()

        system_prompt = (
            "You are a concise telecom analytics assistant. Answer ONLY using the provided "
            "project evidence. Never invent statistics, percentages, or dollar amounts. "
            "Always separate your reply into two clear sections:\n"
            "1. **Retrieved Project Evidence** (strictly facts from the provided context)\n"
            "2. **Operational Considerations (Generated)** (clearly labeled as hypotheses/recommendations, "
            "not proven causal outcomes)."
        )

        user_prompt = f"Question: {question}\n\nProject Evidence:\n{context_text}"
        if customer_context:
            user_prompt += f"\n\nCurrent Customer Prediction Context:\n{json.dumps(customer_context, indent=2)}"

        if provider == "groq" and groq_key and not groq_key.startswith("your_"):
            try:
                from langchain_groq import ChatGroq

                llm = ChatGroq(
                    api_key=groq_key,
                    model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"),
                    temperature=0.1,
                )
                resp = llm.invoke([("system", system_prompt), ("human", user_prompt)])
                return {"mode": "llm_rag_groq", "answer": str(resp.content)}
            except Exception:
                return None

        if openai_key and not openai_key.startswith("your_"):
            try:
                from langchain_openai import ChatOpenAI

                llm = ChatOpenAI(
                    api_key=openai_key,
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    temperature=0.1,
                )
                resp = llm.invoke([("system", system_prompt), ("human", user_prompt)])
                return {"mode": "llm_rag_openai", "answer": str(resp.content)}
            except Exception:
                return None

        return None

    def answer(
        self,
        question: str,
        top_k: int = 3,
        customer_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Answer a user question with strict provenance and clear separation of facts vs recommendations."""
        retrieved = self.retrieve(question, top_k=top_k)
        context_blocks = [
            f"[{r['title']} | source: {r['source']}]\n{r['content']}" for r in retrieved
        ]
        context_text = "\n\n".join(context_blocks) if context_blocks else "No project artifacts indexed yet."

        llm_result = self._try_llm_synthesis(question, context_text, customer_context=customer_context)
        if llm_result is not None:
            return {
                "mode": llm_result["mode"],
                "answer": llm_result["answer"],
                "retrieved_chunks": retrieved,
            }

        # Deterministic, zero-key grounded synthesis
        evidence_lines = []
        for r in retrieved:
            evidence_lines.append(f"**{r['title']}** (`{r['source']}`)\n{r['content']}")

        if customer_context:
            exp = customer_context.get("explanation", {})
            inc = ", ".join(exp.get("top_risk_increasing_factors", [])) or "N/A"
            dec = ", ".join(exp.get("top_risk_decreasing_factors", [])) or "N/A"
            evidence_lines.insert(
                0,
                f"**Active Customer Prediction Context**\n"
                f"- Predicted Churn Probability: {customer_context.get('churn_probability', 0):.1%} "
                f"({customer_context.get('risk_category', 'Unknown')} Risk)\n"
                f"- Top Risk-Increasing SHAP Factors: {inc}\n"
                f"- Top Risk-Decreasing SHAP Factors: {dec}",
            )

        considerations = [
            "Prioritize retention campaigns on high-risk segments (such as month-to-month, early-tenure, "
            "or electronic-check accounts) rather than blanket discounting.",
            "Validate any retention incentive (e.g., contract migration offer or support bundle trial) "
            "through an A/B holdout test before rolling out broadly, since historical model correlations "
            "do not guarantee causal uplift.",
            "Tune the operational decision threshold based on the company's actual outreach cost vs. "
            "customer lifetime value rather than defaulting to 0.50.",
        ]

        answer_md = (
            "### Retrieved Project Evidence\n"
            + "\n\n".join(evidence_lines)
            + "\n\n### Operational Considerations (Generated — Not Proven Causal Outcomes)\n"
            + "\n".join(f"- {c}" for c in considerations)
            + "\n\n*Mode: Local Grounded Retrieval (no external LLM API key required). "
            "Configure `GROQ_API_KEY` or `OPENAI_API_KEY` in `.env` and install `requirements-rag.txt` "
            "to enable conversational LLM synthesis.*"
        )

        return {
            "mode": "local_grounded_rag",
            "answer": answer_md,
            "retrieved_chunks": retrieved,
        }
