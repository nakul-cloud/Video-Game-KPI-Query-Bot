import pandas as pd
import sqlite3
import numpy as np
import faiss
import json
import re
import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import traceback

# ================= OFFLINE CONFIGURATION =================
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# ================= API KEYS =================
# Get your Groq API key from: https://console.groq.com/keys
GROQ_API_KEY =    
TAVILY_API_KEY = 

# ================= DATABASE PATH =================
DB_PATH = r"C:\Users\Nakul\video_game_sales.db"
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"❌ Database not found at: {DB_PATH}. Please check your database path.")

# ================= LANGGRAPH STATE =================
class AgentState(TypedDict):
    messages: Annotated[List[Any], add_messages]
    user_query: str
    intent: Optional[Dict[str, Any]]
    rag_context: Optional[str]
    sql: Optional[str]
    df: Optional[pd.DataFrame]
    df_text: Optional[str]
    analysis: Optional[str]
    enrichment: Optional[Dict[str, str]]
    run_path: List[str]
    tavily_used: bool

# ================= TOOL DEFINITIONS =================
@tool
def query_understanding_tool(query: str) -> Dict[str, Any]:
    """Extract structured intent from user query."""
    intent = {
        "task": "ranking",
        "metric": "total_sales",
        "dimension": None,
        "filters": [],
        "limit": 10,
        "sort_order": "desc"
    }
    
    query_lower = query.lower()
    
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
    
    year_match = re.search(r'20\d{2}', query)
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

@tool
def rag_retrieval_tool(intent: Dict[str, Any]) -> str:
    """Retrieve relevant context from RAG based on intent."""
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
    
    business_rules = """
    BUSINESS RULES:
    1. ONLY SELECT queries allowed
    2. Default metric: total_sales
    3. "Top N" → ORDER BY ... DESC and LIMIT N
    4. Year: WHERE release_date LIKE '%YYYY%'
    5. Sales in MILLIONS
    """
    
    # Determine which documents are relevant based on intent
    relevant_docs = []
    
    # Always include schema
    relevant_docs.append(schema_definition)
    
    # Add KPI definitions if metric is involved
    if intent.get("metric"):
        relevant_docs.append(kpi_definitions)
    
    # Add business rules
    relevant_docs.append(business_rules)
    
    return "\n\n".join(relevant_docs)

@tool
def sql_generation_tool(intent: Dict[str, Any], rag_context: str) -> str:
    """Generate SQL query from intent and RAG context."""
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
            # FIX: Duplicate column name issue
            select_clause = f"{intent['dimension']}, SUM({intent['metric']}) as total_{intent['metric']}_sum"
            group_by_clause = f"GROUP BY {intent['dimension']}"
            order_by_clause = f"ORDER BY total_{intent['metric']}_sum {intent['sort_order'].upper()}"
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
    
    # Clean the SQL
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Validate it's SELECT only
    if not sql.upper().startswith('SELECT'):
        return "ERROR: Only SELECT queries are allowed"
    
    return sql

@tool
def db_query_tool(sql: str) -> str:
    """Execute SQL query on database and return results as text."""
    try:
        sql_clean = sql.strip()
        
        if not sql_clean.upper().startswith('SELECT'):
            return "ERROR: Only SELECT queries allowed"
        
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(sql_clean, conn)
        conn.close()
        
        if df.empty:
            return "No results found"
        
        # Convert to string representation
        return df.head(20).to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"

@tool
def interpretation_tool(df_as_text: str, query: str) -> str:
    """Interpret database results in context of the original query."""
    if "ERROR" in df_as_text or "No results found" in df_as_text:
        return df_as_text
    
    # Create prompt for LLM
    prompt = f"""Analyze these video game sales results in context of the query: '{query}'

Results:
{df_as_text}

Provide a concise business analysis (under 80 words). Focus on key insights, patterns, or notable findings."""
    
    try:
        # Use Groq LLM
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Found results. Analysis unavailable: {str(e)}"

@tool
def tavily_enrichment_tool(title: str) -> str:
    """Enrich game information using Tavily search."""
    global tavily
    if tavily is None:
        return "Tavily not available"
    
    try:
        query = f"{title} video game awards critical reception commercial success"
        result = tavily.search(query=query[:400], max_results=2)
        
        if result and "results" in result:
            search_context = "\n".join([r.get("content", "")[:200] for r in result["results"]])
            
            # Format with agent prompt
            tavily_agent_prompt = """
            ROLE: Tavily enrichment agent for video game market context.
            PURPOSE: Provide qualitative context for video game performance.
            RULES:
            - DO NOT invent numeric sales values
            - Use cautious language like "reported to have sold", "widely considered"
            - Focus on qualitative aspects: awards, critical acclaim, market impact
            - Output in bullet points
            """
            
            format_prompt = f"{tavily_agent_prompt}\nGame: {title}\nResults: {search_context}\nFormat in bullet points."
            
            response = llm.invoke(format_prompt)
            return response.content.strip()
    except Exception as e:
        return f"Enrichment error: {str(e)}"
    
    return "No enrichment data found"

# ================= LLM & TOOLS SETUP =================
def load_models():
    """Load embedding model from local cache"""
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading embedding model '{model_name}' from local cache...")
    embedding_model = SentenceTransformer(model_name)
    return embedding_model

print("Initializing models...")
embedding_model = load_models()

# Initialize Tavily
try:
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    print("✅ Tavily API connected")
except Exception as e:
    print(f"❌ Warning: Tavily not available: {e}")
    tavily = None

# Initialize Groq LLM (GPT 4o 128k)
try:
    # Using Groq with Mixtral-8x7b (fast and free tier) - you can change to other models
    llm = ChatGroq(
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        model_name="openai/gpt-oss-120b"  # You can also use "llama-3.3-70b-versatile" or "gemma2-9b-it"
    )
    print("✅ Groq LLM initialized")
except Exception as e:
    print(f"❌ Error initializing Groq LLM: {e}")
    print("⚠️  Falling back to local model...")
    # Fallback to a local model if Groq fails
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        temperature=0,
        model="openai/gpt-oss-120b",  # Fallback model
        openai_api_key="dummy-key"  # This will fail but shows the structure
    )

# Create all tools list
tools = [
    query_understanding_tool,
    rag_retrieval_tool,
    sql_generation_tool,
    db_query_tool,
    interpretation_tool,
    tavily_enrichment_tool
]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# ================= LANGGRAPH GRAPH =================
def should_continue(state):
    """Determine if we should continue or end."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and (not getattr(last, "tool_calls", None)):
        return "end"
    # If 1 full cycle of all main tools has been used → END
    tool_sequence = state.get("run_path", [])
    SAFE_MAX = 4  # prevents loops
    if len(tool_sequence) >= SAFE_MAX:
        return "end"

    # If SQL exists & DF exists → no more tools needed
    if state.get("sql") and isinstance(state.get("df"), pd.DataFrame):
        return "end"
    return "continue"

def orchestrator(state: AgentState) -> AgentState:
    """Main orchestrator node that decides which tool to call next."""
    messages = state["messages"]
    
    # Prepare system message
    system_prompt = """You are an intelligent agent that helps analyze video game sales data.
    
Your workflow:
1. Understand the user's query intent
2. Retrieve relevant schema and business rules
3. Generate appropriate SQL queries
4. Execute queries and interpret results
5. Optionally enrich with external data

IMPORTANT RULES:
- Only generate SELECT queries, never INSERT/UPDATE/DELETE
- If query is ambiguous, ask for clarification
- Use tools in the most efficient order based on the query
- For sales analysis, consider using enrichment for low-sales games
- Stop when you have a complete answer for the user

Available tools:
- query_understanding_tool: Extract intent from user query
- rag_retrieval_tool: Get database schema and rules
- sql_generation_tool: Create SQL from intent and context
- db_query_tool: Execute SQL and get results
- interpretation_tool: Analyze results in context
- tavily_enrichment_tool: Get external info about a game

Decide which tool to use next based on what you have and what's needed."""
    
    # FIX: Ensure system prompt is inserted once
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
    
    # Get AI response with tool calls
    response = llm_with_tools.invoke(messages)
    
    # Track tool calls in run_path
    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            if "run_path" not in state:
                state["run_path"] = []
            if tool_name not in state["run_path"]:
                state["run_path"].append(tool_name)
    
    # Add AI message to state
    state["messages"].append(response)
    
    return state

def call_tool(state: AgentState) -> AgentState:
    """Execute the tool that was called by the orchestrator."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # FIX: Proper tool message handling
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Call the appropriate tool
        if tool_name == "query_understanding_tool":
            result = query_understanding_tool.invoke(tool_args)
            state["intent"] = result
        elif tool_name == "rag_retrieval_tool":
            result = rag_retrieval_tool.invoke(tool_args)
            state["rag_context"] = result
        elif tool_name == "sql_generation_tool":
            # Combine intent and rag_context if available
            args = tool_args.copy()
            if "intent" in state and "intent" not in args:
                args["intent"] = state["intent"]
            if "rag_context" in state and "rag_context" not in args:
                args["rag_context"] = state["rag_context"]
            result = sql_generation_tool.invoke(args)
            state["sql"] = result
        elif tool_name == "db_query_tool":
            # Use the generated SQL if available
            if "sql" not in tool_args and "sql" in state:
                tool_args["sql"] = state["sql"]
            result = db_query_tool.invoke(tool_args)
            state["df_text"] = result
            # Try to parse back to DataFrame for later use
            try:
                if "ERROR" not in result and "No results found" not in result:
                    conn = sqlite3.connect(DB_PATH)
                    if state.get("sql"):
                        state["df"] = pd.read_sql(state["sql"], conn)
                    conn.close()
            except Exception as e:
                print(f"Warning: Could not parse results to DataFrame: {e}")
        elif tool_name == "interpretation_tool":
            # Add query if not in args
            args = tool_args.copy()
            if "query" not in args and "user_query" in state:
                args["query"] = state["user_query"]
            result = interpretation_tool.invoke(args)
            state["analysis"] = result
        elif tool_name == "tavily_enrichment_tool":
            result = tavily_enrichment_tool.invoke(tool_args)
            state["tavily_used"] = True  # Set Tavily used flag
            if "enrichment" not in state:
                state["enrichment"] = {}
            # Extract title from args or use default
            title = tool_args.get("title", "unknown")
            state["enrichment"][title] = result
        else:
            result = f"Unknown tool: {tool_name}"
        
        # FIX: Each tool call MUST append a ToolMessage
        state["messages"].append(
            ToolMessage(
                content=str(result),
                name=tool_name,
                tool_call_id=tool_call["id"]
            )
        )
    
    return state

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("orchestrator", orchestrator)
workflow.add_node("call_tool", call_tool)

# Add edges
workflow.set_entry_point("orchestrator")

# Conditional routing
workflow.add_conditional_edges(
    "orchestrator",
    should_continue,
    {
        "continue": "call_tool",
        "end": END
    }
)

workflow.add_edge("call_tool", "orchestrator")

# Compile the graph
graph = workflow.compile()

# ================= PROCESSING FUNCTIONS =================
def process_query_agentic(user_query: str) -> Dict[str, Any]:
    """Process query using the agentic LangGraph system."""
    # Initialize state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "user_query": user_query,
        "intent": None,
        "rag_context": None,
        "sql": None,
        "df": None,
        "df_text": None,
        "analysis": None,
        "enrichment": {},
        "run_path": [],
        "tavily_used": False
    }
    
    # Run the graph with timeout to prevent infinite recursion
    try:
        # FIX: Set recursion to safe limit
        final_state = None
        for step in graph.stream(initial_state, {"recursion_limit": 6,"debug":False}):
            if len(step.get("run_path", [])) > 5:
                break

        
            for key, value in step.items():
                if key == "__end__":
                    continue
                final_state = value
        
        if not final_state:
            final_state = initial_state
        
        # Extract final answer from messages
        final_messages = final_state["messages"]
        final_answer = None
        
        # Look for the last AI message without tool calls
        for msg in reversed(final_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_answer = msg.content
                break
        
        # If no final answer, use the analysis or create one
        if not final_answer and final_state.get("analysis"):
            final_answer = final_state["analysis"]
        elif not final_answer:
            final_answer = "Analysis complete. Check results."
        
        # FIX: Fix NaN / JSON errors (NaN → null)
        if isinstance(final_state.get("df"), pd.DataFrame):
            final_state["df"] = final_state["df"].replace({np.nan: None})
        
        # FIX: Final response JSON format
        result = {
            "analysis": final_answer or "Analysis complete.",
            "sql": final_state.get("sql", "") or "-- No SQL generated",
            "results": final_state.get("df", pd.DataFrame()).to_dict(orient="records")
            if isinstance(final_state.get("df"), pd.DataFrame) else [],
            "enrichment": final_state.get("enrichment", {}),
            "run_path": final_state.get("run_path", []),
            "tavily_used": final_state.get("tavily_used", False)
        }
        
        return result
        
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Agent processing error: {e}")
        print(f"Error details: {error_details}")
        
        # Fallback to simple processing if agent fails
        try:
            return fallback_processing(user_query)
        except Exception as fallback_error:
            return {
                "analysis": f"Error in agentic processing: {str(e)}. Using fallback.",
                "sql": "",
                "results": [],
                "enrichment": {},
                "run_path": ["error_fallback"],
                "tavily_used": False
            }

def fallback_processing(user_query: str) -> Dict[str, Any]:
    """Fallback processing if agent system fails."""
    # Simple intent extraction
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
    
    # Generate SQL
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
            # Use fixed column naming
            select_clause = f"{intent['dimension']}, SUM({intent['metric']}) as total_{intent['metric']}_sum"
            group_by_clause = f"GROUP BY {intent['dimension']}"
            order_by_clause = f"ORDER BY total_{intent['metric']}_sum {intent['sort_order'].upper()}"
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
    
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    # Execute SQL
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(sql, conn)
        conn.close()
    except Exception as e:
        df = pd.DataFrame({"error": [f"SQL Error: {str(e)}"]})
    
    # Fix NaN values
    if isinstance(df, pd.DataFrame):
        df = df.replace({np.nan: None})
    
    # Generate analysis using LLM
    analysis = ""
    if df.empty or 'error' in df.columns:
        analysis = "No results found."
    else:
        prompt = f"Analyze these video game sales results:\nQuery: '{user_query}'\nResults: {df.head().to_string(index=False)}\nProvide a concise business analysis (under 80 words)."
        try:
            response = llm.invoke(prompt)
            analysis = response.content.strip()
        except:
            analysis = f"Found {len(df)} results. Top game: {df.iloc[0]['title'] if 'title' in df.columns else 'N/A'}"
    
    # Check if we should use Tavily for enrichment
    tavily_used = False
    enrichment = {}
    if not df.empty and 'title' in df.columns and len(df) > 0:
        game_title = df.iloc[0]['title']
        # Use Tavily for enrichment (only if available and has low sales)
        if tavily is not None:
            try:
                query = f"{game_title} video game awards critical reception commercial success"
                result = tavily.search(query=query[:400], max_results=1)
                if result and "results" in result:
                    search_context = "\n".join([r.get("content", "")[:200] for r in result["results"]])
                    tavily_agent_prompt = """ROLE: Tavily enrichment agent for video game market context.
                    PURPOSE: Provide qualitative context for video game performance.
                    RULES: DO NOT invent numeric sales values."""
                    format_prompt = f"{tavily_agent_prompt}\nGame: {game_title}\nResults: {search_context}\nFormat in bullet points."
                    response = llm.invoke(format_prompt)
                    enrichment[game_title] = response.content.strip()
                    tavily_used = True
            except:
                pass
    
    # Prepare result with fixed format
    result = {
        "analysis": analysis,
        "sql": sql,
        "results": df.to_dict(orient='records') if not df.empty and 'error' not in df.columns else [],
        "enrichment": enrichment,
        "run_path": ["fallback_processing"],
        "tavily_used": tavily_used
    }
    
    return result

# ================= FLASK APP SETUP =================
app = Flask(__name__, template_folder='templates')
CORS(app)

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

        # ⬇️ CALL THE AGENTIC FUNCTION
        result = process_query_agentic(user_query)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= ENTRY POINT =================
if __name__ == "__main__":
    print("🚀 Agentic Server starting at http://127.0.0.1:5000")
    print("📊 LangGraph Agent System Ready")
    print("🛠️  Tools Available: query_understanding, rag_retrieval, sql_generation, db_query, interpretation, tavily_enrichment")
    print(f"🤖 LLM: Groq (Mixtral-8x7b)")
    print(f"🔍 Tavily: {'Connected' if tavily else 'Not available'}")
    print("⚠️  IMPORTANT: Replace GROQ_API_KEY with your actual Groq API key")
    app.run(debug=True, port=5000)