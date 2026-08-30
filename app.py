"""
app.py
Step 3 of the pipeline: a simple Streamlit chat interface on top of
rag_chain.py, so the project is a demoable app rather than just a script.

Run with:
    streamlit run app.py
"""

import streamlit as st
from rag_chain import load_vectorstore, get_answer

st.set_page_config(page_title="Company Docs Q&A", page_icon="📄")
st.title("📄 Company Docs Q&A (RAG)")
st.caption("Ask questions about your uploaded documentation. Answers are "
           "grounded in retrieved chunks, with sources shown below each reply.")

@st.cache_resource
def get_vectorstore():
    return load_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    vectorstore = get_vectorstore()
except Exception as exc:
    st.error("Chatbot setup is incomplete.")
    st.info("Add PDFs to data/, run python ingest.py, and set ANTHROPIC_API_KEY in .env.")
    st.exception(exc)
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = get_answer(question, vectorstore)
        st.markdown(result["answer"])
        with st.expander(f"Sources ({len(result['sources'])})"):
            for i, src in enumerate(result["sources"], 1):
                st.markdown(f"**{i}.** page {src['page']} — {src['source']}")
                st.caption(src["snippet"] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )
