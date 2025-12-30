# 🎮 KPI Query Bot — Agentic RAG with LangGraph & Groq LLM

## 📌 Overview
**KPI Query Bot** is an **Agentic AI system** that understands natural language queries and autonomously:

- Extracts business intent  
- Generates SQL queries  
- Executes them on a database  
- Interprets results as KPI insights  
- Enriches answers when needed  

It is designed for **business intelligence** and currently uses a **Video Game Sales** dataset.

🧠 Powered by:
- **LangGraph** — agent orchestration & tool routing  
- **Groq LLM** — fast reasoning + tool calling  
- **SQLite** — KPI data source  
- **Flask** — frontend bridge/API  
- **Autonomous Agents** — decide next actions, not rules  

---

## 🚀 Features

| Capability | Description |
|------------|--------------|
| 🧠 Agentic RAG | LLM chooses tools & flow based on query |
| 🤖 Tool Calling | Each step handled by specialized tools |
| 🎯 Intent Extraction | Converts user query → structured intent |
| 🔍 Dynamic SQL | Safe SELECT-only query generation |
| 📊 KPI Insights | Summaries with business context |
| 🌍 Enrichment | (Optional) Market context via Tavily |
| 🖥️ Full Stack | Flask API + Web UI |

---

## 📂 Tech Stack

| Layer | Technology |
|-------|-------------|
| UI / API | Flask + HTML |
| Agent Orchestration | **LangGraph** |
| LLM | **Groq Mixtral-8x7B** or Llama-3.3-70B |
| Tools | LangChain Tool Calling |
| DB | SQLite |
| RAG Context | In-code schema + KPI definitions |

👉 **No embeddings / vector database for now** *(future scope)*

---

## 🧠 System Architecture

```mermaid
flowchart LR
    A[User Query] --> B[Flask Frontend]
    B --> C[API /process_query_agentic]
    C --> D[LangGraph Orchestrator]

    subgraph Tools
        T1[Query Understanding Tool]
        T2[RAG Retrieval Tool]
        T3[SQL Generation Tool]
        T4[DB Query Tool]
        T5[Interpretation Tool]
        T6[Tavily Enrichment Tool]
    end

    D -->|Decides next step| Tools
    T3 --> DB[(SQLite Database)]
    DB --> T4
    T4 --> T5 --> D
    D -->|No tools left| E[Final Response]

    E --> F[Flask JSON Response]
    F --> G[Browser UI KPI Output]
