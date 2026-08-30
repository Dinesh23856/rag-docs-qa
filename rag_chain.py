"""
rag_chain.py
Step 2 of the pipeline: given a user question, retrieve the most relevant
chunks from the vector store and ask the LLM to answer using only that
context (this is what makes it "retrieval-augmented" instead of the model
just guessing from its own memory).

Can be run directly for a quick CLI test:
    python rag_chain.py
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

import config

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a
company's internal documentation. Use ONLY the context below to answer the
question. If the answer isn't in the context, say you don't have that
information — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


def validate_configuration():
    if not config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")
    if not config.VECTOR_DB_DIR.exists():
        raise FileNotFoundError("Vector database not found. Run python ingest.py first.")


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory=str(config.VECTOR_DB_DIR),
        embedding_function=embeddings,
    )


def get_answer(question: str, vectorstore=None) -> dict:
    """Retrieve relevant chunks and generate a grounded answer.

    Returns a dict with the answer text and the source chunks used,
    so the UI can show citations.
    """
    validate_configuration()

    if vectorstore is None:
        vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatAnthropic(
        model=config.LLM_MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        temperature=0,
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})

    sources = [
        {
            "page": doc.metadata.get("page"),
            "source": doc.metadata.get("source"),
            "snippet": doc.page_content[:200],
        }
        for doc in relevant_docs
    ]

    return {"answer": response.content, "sources": sources}


if __name__ == "__main__":
    vs = load_vectorstore()
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() == "quit":
            break
        result = get_answer(q, vs)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources used: {len(result['sources'])} chunk(s)")
