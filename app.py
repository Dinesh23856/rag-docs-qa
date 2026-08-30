"""
app.py
Step 3 of the pipeline: a simple Streamlit chat interface on top of
rag_chain.py, so the project is a demoable app rather than just a script.

Includes an in-app PDF uploader so documents can be added without
touching GitHub or a terminal — upload files, tap "Build knowledge
base", and the app ingests them itself.

Run with:
    streamlit run app.py
"""

import streamlit as st

import config
from ingest import split_documents, build_vectorstore
from rag_chain import load_vectorstore, get_answer
from langchain_community.document_loaders import PyPDFLoader

st.set_page_config(page_title="Company Docs Q&A", page_icon="📄")
st.title("📄 Company Docs Q&A (RAG)")
st.caption("Ask questions about your uploaded documentation. Answers are "
           "grounded in retrieved chunks, with sources shown below each reply.")


def index_exists() -> bool:
    return config.VECTOR_DB_DIR.exists() and any(config.VECTOR_DB_DIR.iterdir())


with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    if st.button("Build knowledge base", disabled=not uploaded_files):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        with st.spinner("Saving files..."):
            for f in uploaded_files:
                (config.DATA_DIR / f.name).write_bytes(f.getbuffer())

        with st.spinner("Reading PDFs and splitting into chunks..."):
            documents = []
            for f in uploaded_files:
                loader = PyPDFLoader(str(config.DATA_DIR / f.name))
                documents.extend(loader.load())
            chunks = split_documents(documents)

        with st.spinner("Embedding and saving to the vector store "
                         "(first run downloads the embedding model, "
                         "this can take a minute)..."):
            build_vectorstore(chunks)

        st.cache_resource.clear()
        st.success(f"Indexed {len(uploaded_files)} file(s). Ready to chat!")
        st.rerun()

    if index_exists():
        st.caption("✅ Knowledge base is built.")
    else:
        st.caption("⚠️ No documents indexed yet — upload PDFs above.")


@st.cache_resource
def get_vectorstore():
    return load_vectorstore()


if not index_exists():
    st.info("👈 Upload one or more PDFs in the sidebar and tap "
            "**Build knowledge base** to get started.")
    st.stop()

try:
    vectorstore = get_vectorstore()
except Exception as exc:
    st.error("Chatbot setup is incomplete.")
    st.info("Set ANTHROPIC_API_KEY in your app's Secrets, then reboot the app.")
    st.exception(exc)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

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