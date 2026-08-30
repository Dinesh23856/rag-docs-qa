# RAG Document Q&A Chatbot

A Retrieval-Augmented Generation chatbot that answers questions from your
own PDF documents, instead of relying on the LLM's general knowledge.

## Files

| File | Purpose |
|---|---|
| `config.py` | All settings (paths, model names, chunk size) in one place |
| `ingest.py` | Loads PDFs → splits into chunks → embeds → saves to vector DB |
| `rag_chain.py` | Retrieves relevant chunks for a question and generates a grounded answer |
| `app.py` | Streamlit chat UI on top of `rag_chain.py` |
| `requirements.txt` | Python dependencies |
| `data/` | Put your source PDFs here |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

## Usage

1. Drop your PDF files into `data/`
2. Build the vector database (run once, or whenever docs change):
   ```bash
   python ingest.py
   ```
3. Either test from the command line:
   ```bash
   python rag_chain.py
   ```
   or launch the chat UI:
   ```bash
   streamlit run app.py
   ```

## How it works (the RAG pipeline)

1. **Ingest** — PDFs are split into ~800-character chunks with overlap so
   context isn't lost at chunk boundaries.
2. **Embed** — each chunk is converted into a vector using a free local
   sentence-transformer model (`all-MiniLM-L6-v2`), no API key needed for
   this part.
3. **Store** — vectors are saved in Chroma, a local vector database.
4. **Retrieve** — when a question comes in, its embedding is compared
   against stored chunk embeddings to find the most relevant ones.
5. **Generate** — the retrieved chunks are inserted into a prompt and sent
   to Claude, which answers using only that context (this is what stops
   the model from hallucinating answers not in your documents).

## Swapping in Pinecone or Milvus

This project uses **Chroma** by default because it's free and needs no
account. To use a hosted vector DB instead, you'd change two functions:

- `build_vectorstore()` in `ingest.py`
- `load_vectorstore()` in `rag_chain.py`

replacing the Chroma calls with the equivalent `Pinecone.from_documents(...)`
or `Milvus.from_documents(...)` calls, plus your API credentials in
`config.py`. The rest of the pipeline (chunking, retrieval, generation)
stays exactly the same — this is the standard way these projects are
structured, so recruiters will recognize the pattern either way.
