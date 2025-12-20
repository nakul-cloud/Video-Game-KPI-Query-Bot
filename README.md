# 🎮 Video Game KPI Query Bot  
**Agentic RAG-Based Video Game Sales Intelligence Platform**

An AI-powered analytics system that allows users to query a video game sales database using **natural language** and receive **accurate KPIs, SQL-backed results, and qualitative market insights**.

This project demonstrates the practical use of **Agentic RAG (Retrieval-Augmented Generation)**, **LLMs (Gemini)**, **SQLite**, and **web enrichment via Tavily** in a real-world analytics scenario.

---

## 🚀 Key Features

- 🔍 **Natural Language → SQL Querying**
- 🧠 **Agentic RAG Architecture**
- 📊 **KPI-driven Analytics** (Sales, Platform, Genre, Publisher, etc.)
- 🗄️ **SQLite Database Backend**
- 🌐 **Tavily-powered Web Enrichment** for missing or sparse sales data
- 🎮 **Gaming-inspired Frontend UI** (Vanilla HTML, CSS, JavaScript)
- 🔐 **Read-only SQL Safety Layer**

---

## 🧠 System Architecture Overview

The system follows a **multi-agent pipeline**:

1. **Query Understanding Agent**
   - Parses user intent (metric, filters, ranking, limits)

2. **RAG Retrieval Module**
   - Retrieves schema, KPI definitions, and business rules
   - Uses FAISS + `all-MiniLM-L6-v2` embeddings

3. **SQL Generation Agent**
   - Converts intent + RAG context into SQLite-compatible SELECT queries

4. **SQL Safety Layer**
   - Ensures only read-only (SELECT) queries are executed

5. **Database Layer**
   - Executes SQL on video game sales dataset

6. **Result Interpretation Agent**
   - Converts raw SQL output into business-friendly insights

7. **Tavily Enrichment Agent**
   - Adds **qualitative market context** (awards, popularity, reception)
   - Never fabricates numeric sales values

---

## 🗂️ Project Structure
KPI_Query_Bot/
│
├── backend/
│ └── app.py # FastAPI backend (Agentic RAG pipeline)
│
├── frontend/
│ └── index.html # Vanilla HTML/CSS/JS UI
│
├── video_game_sales.db # SQLite database
│
└── README.md


---

## 📊 Dataset

- Source: **Kaggle – Video Game Sales Dataset**
- Format: CSV → SQLite
- Key Columns:
  - `title`, `console`, `genre`, `publisher`, `developer`
  - `total_sales`, `na_sales`, `jp_sales`, `pal_sales`
  - `critic_score`, `release_date`

> ⚠️ Some records contain missing sales values — handled via Tavily enrichment.

---

## 🔧 Tech Stack

### Backend
- **Python**
- **FastAPI**
- **SQLite**
- **FAISS**
- **SentenceTransformers (`all-MiniLM-L6-v2`)**
- **Google Gemini LLM**
- **Tavily Search API**

### Frontend
- **HTML**
- **Vanilla CSS**
- **JavaScript (Fetch API)**

---

## ▶️ How to Run the Project

### 1️⃣ Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
