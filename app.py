import os
import logging
import re
import streamlit as st
from typing import List
import sys
from io import StringIO

# --- 1. CONFIGURATION & SETUP (Original Script) ---

# Note: API keys will be set via the Streamlit UI, so initial empty values are fine.
os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""
os.environ["SEC_API_KEY"] = ""
os.environ["SEC_API_USER_AGENT"] = "Aurum youremail@example.com"

# Setup logger to stream to a string buffer for display in Streamlit
log_stream = StringIO()
logging.basicConfig(stream=log_stream, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- IMPORTS (Original Script) ---
try:
    from llama_index.core import VectorStoreIndex, Document, Settings, SimpleDirectoryReader
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.node_parser import SentenceSplitter
    from sec_api import ExtractorApi, QueryApi
    from llama_index.tools.tavily_research import TavilyToolSpec
    from llama_index.core.agent import ReActAgent
    from llama_index.core.llms import ChatMessage
except ImportError as e:
    st.error(f"Failed to import a necessary module. Please run 'pip install llama-index sec-api openai tavily-python'. Error: {e}")
    st.stop()

# --- HELPER & CORE FUNCTIONS (Your Original Script) ---
# All your functions (_get_latest_filing_url, create_sec_filing_query_engine, etc.)
# go here without any changes. They are copied directly.

def _get_latest_filing_url(ticker: str, sec_api_key: str) -> str:
    """
    Fetches the latest 10-K filing URL for a given ticker using the SEC Query API.
    """
    logger.info(f"Fetching latest 10-K URL for {ticker}...")
    query_api = QueryApi(api_key=sec_api_key)
    query = {
        "query": { "query_string": {
            "query": f"ticker:{ticker} AND formType:\"10-K\""
        }},
        "from": "0",
        "size": "1",
        "sort": [{ "filedAt": { "order": "desc" }}]
    }
    response = query_api.get_filings(query)
    if response['filings']:
        url = response['filings'][0]['linkToFilingDetails']
        logger.info(f"Found 10-K URL for {ticker}: {url}")
        return url
    else:
        raise FileNotFoundError(f"Could not find a 10-K filing for ticker: {ticker}")

def create_sec_filing_query_engine(ticker: str) -> QueryEngineTool:
    """Creates a QueryEngine over a specific competitor's 10-K."""
    logger.info(f"Creating SEC Query Engine for {ticker}'s 10-K...")
    try:
        sec_api_key = os.environ.get("SEC_API_KEY")
        if not sec_api_key or sec_api_key == "YOUR_SEC_API_KEY":
            raise ValueError("SEC_API_KEY is not set.")
        filing_url = _get_latest_filing_url(ticker, sec_api_key)
        extractor_api = ExtractorApi(sec_api_key)
        section_text = extractor_api.get_section(filing_url, "7", "text")
        if not section_text.strip():
            raise ValueError(f"Extracted section 'Item 7' for {ticker} is empty.")
        docs = [Document(text=section_text, metadata={"ticker": ticker, "filing_type": "10-K", "section": "Item 7"})]
        index = VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine(similarity_top_k=5)
        return QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=f"sec_filing_tool_{ticker}",
                description=(
                    f"Retrieves financial strategy and discussion from Item 7 of {ticker}'s latest 10-K filing. "
                    f"Input should be a clear question about the company's strategy or financial condition."
                )
            ),
        )
    except Exception as e:
        logger.error(f"CRITICAL FAILURE in creating SEC tool for {ticker}: {e}")
        return None

def _extract_tickers_from_prompt(prompt: str) -> List[str]:
    """Extracts all unique stock tickers enclosed in parentheses."""
    tickers = re.findall(r'\(([A-Z]{1,5})\)', prompt)
    unique_tickers = sorted(list(set(tickers)))
    if unique_tickers:
        logger.info(f"Extracted tickers from prompt: {unique_tickers}")
    else:
        logger.warning("No tickers found in parentheses. Example format: 'Apple (AAPL)'.")
    return unique_tickers

def create_operational_rag_query_engine(upload_dir: str) -> QueryEngineTool:
    """Creates a QueryEngine over operational data uploaded by the user."""
    logger.info(f"Attempting to create RAG Engine from files in '{upload_dir}'...")
    try:
        if not os.path.exists(upload_dir) or not os.listdir(upload_dir):
            raise FileNotFoundError(f"No documents found in the upload directory. Please upload a file.")
        loader = SimpleDirectoryReader(input_dir=upload_dir, required_exts=[".txt", ".pdf", ".docx"])
        docs = loader.load_data()
        if not docs:
            raise FileNotFoundError("Could not load any documents from the uploaded file(s).")
        index = VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine(similarity_top_k=5)
        logger.info("✅ Successfully created RAG Engine from uploaded file.")
    except Exception as e:
        logger.error(f"Failed to create RAG Engine: {e}. Using simulated data as fallback.")
        class OperationalMockQueryEngine:
            def __init__(self, error_message):
                self.error_message = error_message
            def query(self, query_str):
                return f"Could not query operational data. Error: {self.error_message} Query attempt: '{query_str}'"
        query_engine = OperationalMockQueryEngine(error_message=str(e))
    return QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="operational_data_rag",
            description="Retrieves internal operational metrics from an uploaded data file."
        ),
    )

def run_financial_strategist_agent(user_prompt: str, upload_dir: str):
    """Initializes and runs the agent with dynamically created tools."""
    # Reset log stream for each run
    log_stream.truncate(0)
    log_stream.seek(0)
    
    # Use a placeholder for intermediate results
    research_placeholder = st.empty()
    strategy_placeholder = st.empty()
    
    with st.spinner("Agent is running... This may take a few minutes."):
        # --- STEP 1: RESEARCH ---
        research_placeholder.info("🚀 **Step 1: Researcher agent is gathering data...**")
        
        # 1. Create base tools
        operational_rag_tool = create_operational_rag_query_engine(upload_dir)
        tavily_spec = TavilyToolSpec(api_key=os.environ.get("TAVILY_API_KEY"))
        
        # 2. Dynamically create SEC tools
        sec_tools = [tool for ticker in _extract_tickers_from_prompt(user_prompt) if (tool := create_sec_filing_query_engine(ticker=ticker))]
        
        # 3. Assemble all tools
        all_tools = [operational_rag_tool] + tavily_spec.to_tool_list() + sec_tools
        
        # 4. Define and run researcher agent
        researcher_prompt = (
            "You are a world-class financial research assistant. Your ONLY job is to use the provided tools to gather factual data in response to the user's query. "
            "Do not add any analysis or recommendations. Simply execute the tool calls and present the gathered information clearly."
        )
        researcher_agent = ReActAgent.from_tools(tools=all_tools, llm=Settings.llm, system_prompt=researcher_prompt, verbose=True)
        
        # Capture stdout to display agent's thought process
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        research_analysis = researcher_agent.chat(user_prompt)
        
        sys.stdout = old_stdout # Restore stdout
        
        research_placeholder.success("✅ **Step 1: Research Complete!**")
        with st.expander("Show Raw Research Data and Agent Steps"):
            st.text(captured_output.getvalue()) # Show agent's verbose output
            st.markdown("---")
            st.markdown(str(research_analysis))

        # --- STEP 2: STRATEGY ---
        strategy_placeholder.info("🚀 **Step 2: Strategist LLM is generating recommendations...**")
        
        strategist_prompt_template = (
            "You are a world-class financial strategist. Your task is to create a set of actionable recommendations for a company based on the research analysis provided."
            "\n\n**USER'S CORE PROBLEM:**\n{original_query}"
            "\n\n**RESEARCH ANALYSIS:**\n{research_summary}"
            "\n\n**YOUR TASK:**"
            "\nBased on the user's problem and the research analysis, create a final section called 'Strategic Recommendations.' "
            "This section MUST propose a concrete, numbered list of 2-3 strategic actions the user's company should take. "
            "Do not just summarize the data again. Provide a clear, forward-looking strategy."
        )
        final_prompt_str = strategist_prompt_template.format(original_query=user_prompt, research_summary=str(research_analysis))
        messages = [ChatMessage(role="user", content=final_prompt_str)]
        final_strategy_response = Settings.llm.chat(messages)
        
        strategy_placeholder.success("✅ **Final Strategy Generated!**")
        st.markdown("---")
        st.subheader("🏆 Strategic Recommendations")
        st.markdown(str(final_strategy_response))
        
    with st.expander("View Full Agent Logs"):
        st.text(log_stream.getvalue())


# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="Financial Strategist Agent", layout="wide")
st.title("📈 Financial Strategist Agent")

# Sidebar for API keys and file upload
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    sec_api_key = st.text_input("SEC-API.io Key", type="password")
    
    st.header("📄 Operational Data")
    uploaded_file = st.file_uploader(
        "Upload your internal data (.txt, .pdf, .docx)",
        type=['txt', 'pdf', 'docx']
    )

st.info("Enter your query below. Make sure to include company tickers in parentheses, like `(F)` or `(GM)`.")

# Main input for user query
default_query = (
    "Based on my company's recent margin compression, I need a strategic analysis. "
    "Please compare the capital allocation and risk factor strategies of Ford (F) and General Motors (GM) "
    "using their latest 10-K filings. Also, what are the current market trends for EV charging infrastructure?"
)
user_prompt = st.text_area("Your Query:", value=default_query, height=150)

# Run button
if st.button("Generate Strategy", type="primary"):
    # --- Input Validation ---
    if not openai_api_key or not tavily_api_key or not sec_api_key:
        st.error("Please enter all required API keys in the sidebar.")
    elif not user_prompt:
        st.error("Please enter a query.")
    else:
        # --- Setup Environment and Models ---
        try:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["TAVILY_API_KEY"] = tavily_api_key
            os.environ["SEC_API_KEY"] = sec_api_key

            Settings.llm = OpenAI(model="gpt-4o", temperature=0.1)
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
            Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)
            
            # --- Handle File Upload ---
            upload_dir = "uploaded_docs"
            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)
            
            # Clear previous files
            for f in os.listdir(upload_dir):
                os.remove(os.path.join(upload_dir, f))

            if uploaded_file is not None:
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.sidebar.success(f"Uploaded `{uploaded_file.name}`")
            
            # --- Run the Agent ---
            run_financial_strategist_agent(user_prompt, upload_dir)

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"UI-level error: {e}", exc_info=True)
            with st.expander("View Full Error Logs"):
                st.text(log_stream.getvalue())