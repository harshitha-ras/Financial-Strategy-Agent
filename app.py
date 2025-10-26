import os
import logging
import re
import streamlit as st
from typing import List
import sys
from io import StringIO
import calcbench

# --- 1. CONFIGURATION & SETUP (Original Script) ---

# Note: API keys will be set via the Streamlit UI, so initial empty values are fine.
os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""


def setup_calcbench(email, password):
    calcbench.set_credentials(email, password)


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
    from llama_index.tools.tavily_research import TavilyToolSpec
    from llama_index.core.agent import ReActAgent
    from llama_index.core.llms import ChatMessage
except ImportError as e:
    st.error(f"Failed to import a necessary module. Please run 'pip install llama-index sec-api openai tavily-python'. Error: {e}")
    st.stop()

# --- HELPER & CORE FUNCTIONS (Your Original Script) ---
# go here without any changes. They are copied directly.

# SEC API functions removed in favor of Calcbench API

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

def create_calcbench_filing_query_engine(ticker: str) -> QueryEngineTool:
    """Creates a QueryEngine over a competitor's 10-K MD&A using Calcbench."""
    logger.info(f"Creating Calcbench Query Engine for {ticker} 10-K (MD&A)...")
    try:
        # Look up Calcbench company ID
        company_id = calcbench.ticker_lookup(ticker)
        filings = calcbench.filings(company_id=company_id, formType='10-K')
        if not filings:
            raise FileNotFoundError(f"No 10-K filings found for {ticker} via Calcbench.")
        filing = filings[0]
        # Calcbench Item 7 MD&A scraping:
        mda = calcbench.mda_text(filing['filingId'])
        if not mda:
            raise ValueError(f"Item 7 MD&A section not found for {ticker}.")
        docs = [Document(text=mda, metadata={"ticker": ticker, "filing_type": "10-K", "section": "Item 7"})]
        index = VectorStoreIndex.from_documents(docs)
        query_engine = index.as_query_engine(similarity_top_k=5)
        return QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=f"calcbench_filing_tool_{ticker}",
                description=(f"Retrieves MD&A from Item 7 of {ticker}'s 10-K from Calcbench.")
            ),
        )
    except Exception as e:
        logger.error(f"Error creating Calcbench tool for {ticker}: {e}")
        return None


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
        
        # 2. Dynamically create Calcbench tools for each ticker
        calcbench_tools = [tool for ticker in _extract_tickers_from_prompt(user_prompt) if (tool := create_calcbench_filing_query_engine(ticker))]

        
        # 3. Assemble all tools
        all_tools = [operational_rag_tool] + tavily_spec.to_tool_list() + calcbench_tools
        
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

def batch_run_eval(eval_csv_path: str, model_output_csv: str, upload_dir: str):
    import pandas as pd
    eval_df = pd.read_csv(eval_csv_path)
    results = []
    for idx, row in eval_df.iterrows():
        question = row['question']
        context = row['context']  # Pass this (if your agent takes a context arg)
        # You may need to adjust if you use upload files/per-question context

        # Run your agent once per row, without UI (headless)
        answer = run_financial_strategist_agent(question, upload_dir)  # returns agent string output
        
        results.append({'question': question, 'model_output': answer})

    result_df = pd.DataFrame(results)
    result_df.to_csv(model_output_csv, index=False)
    print(f"Saved batch model outputs to {model_output_csv}")



# --- 2. STREAMLIT UI ---

st.set_page_config(page_title="Financial Strategist Agent", layout="wide")
st.title("📈 Financial Strategist Agent")

# Sidebar for API keys and file upload
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    tavily_api_key = st.text_input("Tavily API Key", type="password")
    
    st.header("🔑 Calcbench Credentials")
    calcbench_email = st.text_input("Calcbench Username/Email")
    calcbench_password = st.text_input("Calcbench Password", type="password")
    
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
    if not openai_api_key or not tavily_api_key or not calcbench_email or not calcbench_password:
        st.error("Please enter all required API keys in the sidebar.")
    elif not user_prompt:
        st.error("Please enter a query.")
    else:
        # --- Setup Environment and Models ---
        try:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["TAVILY_API_KEY"] = tavily_api_key
            setup_calcbench(calcbench_email, calcbench_password)

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