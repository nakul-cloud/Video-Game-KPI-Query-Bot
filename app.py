import pandas as pd
import sqlite3
import numpy as np
import faiss
import json
import re
import os
import torch
from sentence_transformers import SentenceTransformer
from google import genai
from tavily import TavilyClient
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# ================= OFFLINE CONFIGURATION =================
# This forces the libraries to use local files and skip internet checks
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# ================= FLASK SETUP =================
app = Flask(__name__)
CORS(app)

# ================= API KEYS =================
GEMINI_API_KEY =  
TAVILY_API_KEY =  

# ================= DATABASE PATH =================
DB_PATH = r"C:\Users\Nakul\video_game_sales.db"

# Check if database exists
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"❌ Database not found at: {DB_PATH}. Please check your database path.")

# ================= LOAD MODELS =================
def load_models():
    """Load Gemini model and embedding model from local cache"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # We use the standard model name; because of the environment variables 
    # and your previous successful run, it will load from your local PC cache.
    model_name = "all-MiniLM-L6-v2"
    
    print(f"Loading embedding model '{model_name}' from local cache...")
    embedding_model = SentenceTransformer(model_name)
    return client, embedding_model

print("Initializing models...")
llm_client, embedding_model = load_models()

# Initialize Tavily
try:
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
except Exception as e:
    print(f"Warning: Tavily not available: {e}")
    tavily = None

# ================= TAVILY AGENT PROMPT =================
TAVILY_AGENT_PROMPT = """
ROLE: Tavily enrichment agent for video game market context.
PURPOSE: Provide qualitative context for video game performance.
RULES:
- DO NOT invent numeric sales values
- Use cautious language like "reported to have sold", "widely considered"
- Focus on qualitative aspects: awards, critical acclaim, market impact
- Output in bullet points
"""

# ================= SCHEMA =================
schema_definition = """
DATABASE SCHEMA DEFINITION
Table: video_game_sales
Columns:
1. img (TEXT) - Game cover image
2. title (TEXT) - Game name
3. console (TEXT) - Platform (PlayStation, Xbox, Switch, PC)
4. genre (TEXT) - Category
5. publisher (TEXT) - Publishing company
6. developer (TEXT) - Development studio
7. critic_score (REAL) - Critic rating (0-100)
8. total_sales (REAL) - Worldwide sales (MILLIONS)
9. na_sales (REAL) - North America sales
10. jp_sales (REAL) - Japan sales
11. pal_sales (REAL) - Europe sales
12. other_sales (REAL) - Other regions sales
13. release_date (TEXT) - Release date
14. last_update (TEXT) - Last update date
"""

# ================= KPI DEFINITIONS =================
kpi_definitions = """
KPI DEFINITIONS:
SALES METRICS:
- Total Sales → total_sales
- North America → na_sales  
- Japan → jp_sales
- Europe → pal_sales
- Other Regions → other_sales

AGGREGATION:
- Top/Best → ORDER BY ... DESC
- Group by Platform → GROUP BY console
- Group by Genre → GROUP BY genre
- Group by Publisher → GROUP BY publisher

TIME FILTERS:
- Year filter → WHERE release_date LIKE '%2023%'
- Recent → ORDER BY release_date DESC

RANKING:
- Always include LIMIT for ranking
"""

# ================= BUSINESS RULES =================
business_rules = """
BUSINESS RULES:
1. ONLY SELECT queries allowed
2. Default metric: total_sales
3. "Top N" → ORDER BY ... DESC and LIMIT N
4. Year: WHERE release_date LIKE '%YYYY%'
5. Sales in MILLIONS
"""

# ================= RAG SETUP =================
def setup_rag():
    rag_documents = [
        {"type": "schema", "content": schema_definition},
        {"type": "kpi", "content": kpi_definitions},
        {"type": "business_rules", "content": business_rules}
    ]

    rag_texts = [doc["content"] for doc in rag_documents]
    embeddings = embedding_model.encode(
        rag_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    id_to_doc = {i: rag_documents[i] for i in range(len(rag_documents))}
    return index, id_to_doc

print("Setting up RAG index...")
index, id_to_doc = setup_rag()

# ================= HELPER FUNCTIONS =================
def clean_sql(sql_text):
    """Clean SQL by removing markdown"""
    sql_text = re.sub(r"```sql", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"```", "", sql_text)
    sql_text = sql_text.strip()
    
    if not sql_text.upper().startswith('SELECT'):
        select_match = re.search(r'(SELECT .*?)(?:;|$)', sql_text, re.IGNORECASE | re.DOTALL)
        if select_match:
            sql_text = select_match.group(1).strip()
        else:
            return ""
    
    if not sql_text.endswith(';'):
        sql_text = sql_text + ';'
    
    return sql_text

def call_gemini(prompt: str) -> str:
    """Call Gemini API"""
    try:
        response = llm_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error calling Gemini: {str(e)}"

def retrieve_rag_context(query):
    """Retrieve relevant context from RAG"""
    q_emb = embedding_model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    _, idx = index.search(np.array([q_emb]), 2)
    
    contexts = []
    for i in idx[0]:
        if i < len(id_to_doc):
            contexts.append(id_to_doc[i]["content"])
    
    return "\n\n".join(contexts)

def execute_sql(sql):
    """Execute SQL query"""
    try:
        sql_clean = sql.strip()
        
        if not sql_clean.upper().startswith('SELECT'):
            return pd.DataFrame({"error": [f"Only SELECT queries allowed"]})
        
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(sql_clean, conn)
        conn.close()
        
        return df
    except Exception as e:
        return pd.DataFrame({"error": [f"SQL Error: {str(e)}"]})

# ================= AGENTS =================
def query_understanding_agent(user_query):
    """Extract structured intent"""
    intent = {
        "task": "ranking",
        "metric": "total_sales",
        "dimension": None,
        "filters": [],
        "limit": 10,
        "sort_order": "desc"
    }
    
    query_lower = user_query.lower()
    
    if "critic" in query_lower or "score" in query_lower:
        intent["metric"] = "critic_score"
    elif "north america" in query_lower or "na" in query_lower:
        intent["metric"] = "na_sales"
    elif "japan" in query_lower or "jp" in query_lower:
        intent["metric"] = "jp_sales"
    elif "europe" in query_lower or "pal" in query_lower:
        intent["metric"] = "pal_sales"
    
    if "console" in query_lower or "platform" in query_lower:
        intent["dimension"] = "console"
        intent["task"] = "aggregation"
    elif "genre" in query_lower:
        intent["dimension"] = "genre"
        intent["task"] = "aggregation"
    elif "publisher" in query_lower:
        intent["dimension"] = "publisher"
        intent["task"] = "aggregation"
    elif "developer" in query_lower:
        intent["dimension"] = "developer"
        intent["task"] = "aggregation"
    
    year_match = re.search(r'20\d{2}', user_query)
    if year_match:
        year = year_match.group(0)
        intent["filters"].append(f"year={year}")
    
    if "playstation" in query_lower:
        intent["filters"].append("console='PlayStation'")
    elif "xbox" in query_lower:
        intent["filters"].append("console='Xbox'")
    elif "nintendo" in query_lower or "switch" in query_lower:
        intent["filters"].append("console='Nintendo'")
    elif "pc" in query_lower:
        intent["filters"].append("console='PC'")
    
    limit_match = re.search(r'top\s+(\d+)', query_lower) or re.search(r'(\d+)\s+games', query_lower)
    if limit_match:
        intent["limit"] = int(limit_match.group(1))
    
    if "worst" in query_lower or "lowest" in query_lower:
        intent["sort_order"] = "asc"
    
    return intent

def sql_generation_agent(intent, rag_context):
    """Generate SQL query"""
    where_parts = []
    for filt in intent["filters"]:
        if '=' in filt:
            key, value = filt.split('=', 1)
            if key == 'year':
                where_parts.append(f"release_date LIKE '%{value}%'")
            else:
                where_parts.append(f"{key} = '{value}'")
    
    where_clause = "WHERE " + " AND ".join(where_parts) if where_parts else ""
    
    if intent["task"] == "aggregation" and intent["dimension"]:
        if intent["metric"] == "critic_score":
            select_clause = f"{intent['dimension']}, AVG({intent['metric']}) as avg_score"
            group_by_clause = f"GROUP BY {intent['dimension']}"
            order_by_clause = f"ORDER BY avg_score {intent['sort_order'].upper()}"
        else:
            select_clause = f"{intent['dimension']}, SUM({intent['metric']}) as total_sales"
            group_by_clause = f"GROUP BY {intent['dimension']}"
            order_by_clause = f"ORDER BY total_sales {intent['sort_order'].upper()}"
    else:
        select_clause = "title, console, genre, publisher, total_sales, critic_score"
        if intent["metric"] != "total_sales":
            select_clause += f", {intent['metric']}"
        group_by_clause = ""
        order_by_clause = f"ORDER BY {intent['metric']} {intent['sort_order'].upper()}"
    
    limit_clause = f"LIMIT {intent['limit']}" if intent.get("limit") else ""
    
    sql = f"""
    SELECT {select_clause}
    FROM video_game_sales
    {where_clause}
    {group_by_clause}
    {order_by_clause}
    {limit_clause};
    """
    
    return re.sub(r'\s+', ' ', sql).strip()

def result_interpretation_agent(df, user_query, sql):
    if df.empty or 'error' in df.columns:
        return "No results found."
    
    prompt = f"Analyze these video game sales results:\nQuery: '{user_query}'\nResults: {df.head().to_string(index=False)}\nProvide a concise business analysis (under 80 words)."
    
    try:
        return call_gemini(prompt)
    except:
        return f"Found {len(df)} results. Top game: {df.iloc[0]['title'] if 'title' in df.columns else 'N/A'}"

def tavily_enrichment_agent(game_title):
    if tavily is None: return None
    try:
        query = f"{game_title} video game awards critical reception commercial success"
        result = tavily.search(query=query[:400], max_results=2)
        if result and "results" in result:
            search_context = "\n".join([r.get("content", "")[:200] for r in result["results"]])
            format_prompt = f"{TAVILY_AGENT_PROMPT}\nGame: {game_title}\nResults: {search_context}\nFormat in bullet points."
            return call_gemini(format_prompt)
    except:
        return None

# ================= MAIN PROCESSING =================
def process_query(user_query: str):
    result = {"intent": None, "sql": "", "data": None, "analysis": "", "enrichment": {}}
    try:
        intent = query_understanding_agent(user_query)
        result["intent"] = intent
        rag_context = retrieve_rag_context(user_query)
        sql = sql_generation_agent(intent, rag_context)
        result["sql"] = sql
        df = execute_sql(sql)
        result["data"] = df
        
        if not df.empty and 'error' not in df.columns:
            result["analysis"] = result_interpretation_agent(df, user_query, sql)
            if 'title' in df.columns and len(df) > 0:
                game_title = df.iloc[0]['title']
                enrichment = tavily_enrichment_agent(game_title)
                if enrichment: result["enrichment"][game_title] = enrichment
    except Exception as e:
        result["analysis"] = f"Error: {str(e)}"
    return result

# ================= WEB ROUTES =================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        user_query = data.get('query')
        if not user_query:
            return jsonify({"error": "No query provided"}), 400

        result = process_query(user_query)
        
        # Convert DataFrame to JSON for the frontend
        if isinstance(result['data'], pd.DataFrame):
            if 'error' in result['data'].columns:
                return jsonify({"error": result['data']['error'].iloc[0]}), 500
            result['data'] = result['data'].to_dict(orient='records')
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= ENTRY POINT =================
if __name__ == "__main__":
    print("🚀 Server starting at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)