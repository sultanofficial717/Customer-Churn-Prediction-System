"""Page 4: Optional Evidence-Grounded AI & RAG Churn Analytics Assistant."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.assistant import ChurnRAGAssistant

st.set_page_config(page_title="AI Assistant | Churn System", page_icon="🤖", layout="wide")


@st.cache_resource
def load_assistant() -> ChurnRAGAssistant:
    return ChurnRAGAssistant()


st.title("🤖 Churn Analytics AI Assistant (RAG)")
st.caption(
    "Ask questions about customer churn segments, model evaluation metrics, SHAP risk drivers, "
    "or your most recent customer prediction. Retrieved project evidence is strictly separated "
    "from generated operational recommendations."
)

assistant = load_assistant()

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

last_pred = st.session_state.get("last_prediction")
if last_pred:
    st.success(
        f"Active customer prediction loaded in context: **{last_pred['churn_probability']:.1%}** "
        f"churn probability (**{last_pred['risk_category']} Risk**). You can ask why this customer is at risk."
    )
else:
    st.info(
        "Tip: Run a prediction on the **Customer Prediction** page first to ask customer-specific "
        "questions here, or ask general questions about the dataset and model below."
    )

# Starter prompt buttons
st.markdown("**Suggested Questions:**")
sample_questions = [
    "Which customer segment has the highest churn?",
    "What factors contribute most to churn according to SHAP?",
    "Which contract type is associated with lower churn?",
    "How did Logistic Regression, Random Forest, and XGBoost compare?",
    "Why is this particular customer at high risk?",
]
cols = st.columns(len(sample_questions))
selected_sample = None
for col, q in zip(cols, sample_questions):
    if col.button(q, use_container_width=True):
        selected_sample = q

# Render chat history
for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("retrieved_chunks"):
            with st.expander(f"Sources retrieved ({msg.get('mode', 'rag')})"):
                for chunk in msg["retrieved_chunks"]:
                    st.markdown(
                        f"- **{chunk['title']}** (`{chunk['source']}`, relevance={chunk['score']:.3f})"
                    )

user_query = st.chat_input("Ask about churn patterns, model metrics, SHAP insights, or retention strategy...")
prompt = selected_sample or user_query

if prompt:
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = assistant.answer(prompt, top_k=3, customer_context=last_pred)
    with st.chat_message("assistant"):
        st.markdown(response["answer"])
        with st.expander(f"Sources retrieved ({response['mode']})"):
            for chunk in response["retrieved_chunks"]:
                st.markdown(
                    f"- **{chunk['title']}** (`{chunk['source']}`, relevance={chunk['score']:.3f})"
                )

    st.session_state["chat_history"].append(
        {
            "role": "assistant",
            "content": response["answer"],
            "mode": response["mode"],
            "retrieved_chunks": response["retrieved_chunks"],
        }
    )
