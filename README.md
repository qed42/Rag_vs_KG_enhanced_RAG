# RAG vs Knowledge Graph-Enhanced RAG

<p align="center">
  <em>A side-by-side experiment comparing standard vector-based RAG against a Knowledge Graph-enhanced RAG pipeline — on the same documents, the same questions, at the same time.</em>
</p>

## Overview

This project is a Streamlit app that lets you upload documents and ask questions against **two RAG pipelines running in parallel**:

1. **Traditional RAG** — a FAISS vector store + similarity search + LLM answer generation.
2. **KG-Enhanced RAG** — the same vector retrieval, plus a **Neo4j knowledge graph** built from the documents (entities + relationships extracted via an LLM), which is queried alongside the vector context before generating an answer.

Every question you ask is answered by both pipelines, side by side, with response time tracked for each — so you can directly compare retrieval quality, reasoning depth, and latency.

This is a companion implementation to the write-up **["How Knowledge Graphs Take RAG Beyond Retrieval"](https://www.qed42.com/insights/how-knowledge-graphs-take-rag-beyond-retrieval)**, which lays out the theory: plain vector search is fast but struggles with ambiguity and multi-hop reasoning, while knowledge graphs add structured, explainable relationships that ground and connect retrieved facts. This app is the hands-on version of that comparison.

## 🚀 Features

- **Multi-format ingestion** — PDF, DOCX, PPTX, and TXT files via drag-and-drop upload.
- **Dual pipeline execution** — every query runs through both Traditional RAG and KG-Enhanced RAG, with results shown side by side.
- **Automatic knowledge graph construction** — entities and relationships are extracted from document chunks via an LLM and written into Neo4j.
- **Response time tracking** — each answer reports its latency, making the speed/accuracy trade-off visible.
- **Interactive graph visualization** — explore the constructed knowledge graph using a PyVis network view.
- **Persistent chat history** — all question/answer comparisons for the session are kept and displayed for review.

## 🧠 Architecture

```
Documents (PDF/DOCX/PPTX/TXT)
        │
        ▼
 DocumentProcessor → chunks (RecursiveCharacterTextSplitter)
        │
        ├──────────────────────────────┐
        ▼                              ▼
 Traditional RAG                KG-Enhanced RAG
 ─────────────────              ─────────────────
 FAISS vector store      FAISS vector store + Neo4j knowledge graph
 (OpenAIEmbeddings)       (entities/relationships extracted via LLM,
        │                  stored as nodes/edges in Neo4j)
        ▼                              ▼
 RetrievalQA chain         Vector context + graph context merged
        │                  into a single prompt → LLM
        ▼                              ▼
      Answer + latency            Answer + latency
        │                              │
        └──────────────┬───────────────┘
                        ▼
              Side-by-side comparison UI
```

**Key pieces:**

| Component | Role |
|---|---|
| `DocumentProcessor` | Extracts text from PDF/DOCX/PPTX/TXT and splits it into chunks |
| `RAGSystem` (`traditional`) | Embeds chunks into FAISS, answers via a standard `RetrievalQA` chain |
| `RAGSystem` (`kg`) | Same FAISS retrieval, plus LLM-based entity/relationship extraction written into Neo4j, queried alongside vector context |
| `KnowledgeGraphManager` | Owns the Neo4j driver — graph construction, querying, statistics, and PyVis visualization |
| Streamlit UI | File upload, processing status, dual-pane Q&A comparison, chat history, graph stats |

## 🛠️ Tech Stack

- **App framework:** Streamlit
- **Orchestration:** LangChain (`LLMChain`, `RetrievalQA`, `PromptTemplate`)
- **LLM & embeddings:** OpenAI (`OpenAI`, `OpenAIEmbeddings`)
- **Vector store:** FAISS
- **Knowledge graph:** Neo4j (via the `neo4j` Python driver)
- **Graph visualization:** NetworkX + PyVis
- **Document parsing:** PyPDF2, docx2txt, python-pptx

## 📦 Setup

### Prerequisites

- Python 3.9+
- A running Neo4j instance (local or [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/)), with a known URI, username, and password
- An OpenAI API key

### Installation

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
```

### Run

```bash
streamlit run app.py
```

## ▶️ Usage

1. Upload one or more documents (PDF, DOCX, PPTX, or TXT) in the sidebar.
2. Click **🚀 Process Documents** — this builds the FAISS index *and* the Neo4j knowledge graph in the background.
3. Once processing completes, type a question and click **Compare RAG vs KG-RAG**.
4. Review both answers side by side, along with their response times.
5. Scroll down for **chat history** of every comparison run in the session, and (optionally) the knowledge graph summary/visualization.

## 🧪 Experiments & Outcomes

The goal of running documents through both pipelines is to make the trade-offs from the [companion article](https://www.qed42.com/insights/how-knowledge-graphs-take-rag-beyond-retrieval) concrete and measurable, rather than theoretical. Recommended way to use this section: run a fixed set of test questions against a fixed document set through the app, and log the table below.

| Question | Traditional RAG answer | Traditional RAG latency | KG-RAG answer | KG-RAG latency | Notes |
|---|---|---|---|---|---|
| *e.g. "Who leads Project X and which team do they report to?"* | | | | | *Multi-hop questions are where KG-RAG should pull ahead* |
| *e.g. "Summarize the document's main topic"* | | | | | *Simple/single-fact questions are where Traditional RAG should be competitive on speed* |

**What to look for when filling this in**, based on the underlying theory:

- **Latency:** Traditional RAG should consistently be faster — it skips the graph traversal and the extra context-merging step entirely.
- **Multi-hop / relational questions** (e.g., *"how are X and Y connected?"*): KG-RAG should produce more complete, more clearly grounded answers, since it can follow explicit edges between entities rather than relying on embedding similarity alone.
- **Simple factual lookups:** the two pipelines should converge — the graph adds little extra value here, so the latency cost of KG-RAG is the main difference.
- **Ambiguous entity references** (e.g., same name referring to different things in different parts of a document): KG-RAG should resolve these more reliably, since entities are explicit nodes rather than implicit vector positions.
- **Document coverage with no extractable entities** (e.g., narrative prose with few named things): the gap between the two pipelines should shrink, since the knowledge graph has less structure to add.

> Once you've run your own test set, replace the placeholder table above with your actual logged questions, answers, and timings — that becomes the evidence for (or against) using a knowledge graph for your specific document type and query patterns.

## ⚠️ Known Limitations

- Knowledge graph construction makes one LLM call **per chunk** to extract entities/relationships — this is slow and costly on large documents.
- Entity extraction quality depends entirely on the LLM's JSON output being well-formed; malformed responses are skipped with a warning rather than retried.
- The knowledge graph is cleared and rebuilt on every new document upload — there's no incremental/persistent graph across sessions.
- Graph visualization is resource-intensive on large graphs and is best used on smaller test documents.

## 📄 References

- [How Knowledge Graphs Take RAG Beyond Retrieval](https://www.qed42.com/insights/how-knowledge-graphs-take-rag-beyond-retrieval) — the conceptual write-up this project implements.

## 🤝 Contributing

Issues and pull requests are welcome — especially additional test questions/documents for the experiments table above.

## 📄 License

*(Add your license here, e.g., MIT, Apache 2.0.)*
