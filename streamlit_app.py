import streamlit as st
import os
import re
import requests
import pandas as pd
import tempfile
from bs4 import BeautifulSoup
from datetime import datetime
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Any, Dict
import time
from io import StringIO
import sys

# Page configuration
st.set_page_config(
    page_title="🌾 Krishi Samadhan – Farmonomics",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2E8B57;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
.tool-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2E8B57;
    margin: 1rem 0;
}
.thinking-process {
    background: #e3f2fd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1976d2;
}
.confidence-score {
    background: #f3e5f5;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #7b1fa2;
}
.success-metric {
    background: #e8f5e8;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #4caf50;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store_built' not in st.session_state:
    st.session_state.vector_store_built = False
if 'historical_data_loaded' not in st.session_state:
    st.session_state.historical_data_loaded = False
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = {}
if 'agent_ready' not in st.session_state:
    st.session_state.agent_ready = False

# --- Perplexity LLM Wrapper with Tracking ---
class PerplexityLLM(LLM):
    """Custom LangChain LLM wrapper for the Perplexity API with usage tracking."""
    api_key: str
    model: str = "sonar-pro"
    temperature: float = 0.0

    def __init__(self, api_key: str, model: str = "sonar-pro", temperature: float = 0.0, **kwargs):
        super().__init__(api_key=api_key, model=model, temperature=temperature, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "perplexity"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        try:
            start_time = time.time()
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=data,
                timeout=45
            )
            response.raise_for_status()
            result = response.json()
            end_time = time.time()
            
            # Store metrics in session state
            if 'api_metrics' not in st.session_state:
                st.session_state.api_metrics = []
            
            st.session_state.api_metrics.append({
                'response_time': end_time - start_time,
                'prompt_length': len(prompt),
                'response_length': len(result["choices"][0]["message"]["content"]),
                'timestamp': datetime.now()
            })
            
            return result["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            return f"API Error: {e}"
        except Exception as e:
            return f"An unexpected error occurred during the API call: {e}"

# --- Captured Output Context Manager ---
class CaptureOutput:
    def __init__(self):
        self.captured_output = StringIO()
        self.original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self.captured_output
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout

    def get_output(self):
        return self.captured_output.getvalue()

# --- Tool 1: Latest Mandi Price Scraper ---
@tool
def get_latest_mandi_price() -> str:
    """
    Fetches the latest 'Dara' variety wheat price from mandiprices.com for Kanpur.
    Returns a string with the findings or an error message.
    """
    url = "https://www.mandiprices.com/wheat-price/uttar-pradesh/kanpur-grain/dara.html"
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response = session.get(url, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        page_text = soup.get_text()

        max_price_match = re.search(r'Maximum\s+Price\s*:?\s*₹?\s*([\d,]+)', page_text, re.IGNORECASE)
        avg_price_match = re.search(r'Average\s+Price\s*:?\s*₹?\s*([\d,]+)', page_text, re.IGNORECASE)

        max_price = int(max_price_match.group(1).replace(',', '')) if max_price_match else "Not found"
        avg_price = int(avg_price_match.group(1).replace(',', '')) if avg_price_match else "Not found"

        if max_price == "Not found" and avg_price == "Not found":
            return "Could not find the latest wheat price for Kanpur. The website structure may have changed."

        return f"Today's wheat price in Kanpur: Maximum Price is ₹{max_price}, Average Price is ₹{avg_price}."

    except requests.RequestException as e:
        return f"Error fetching latest price data: {e}"
    except Exception as e:
        return f"An unexpected error occurred while fetching the latest price: {e}"

# --- Tool 2: Historical Price Analysis Agent ---
@tool
def query_historical_prices(query: str) -> str:
    """
    Answers questions about historical wheat prices from 2020-2024 using uploaded CSV data.
    Use this for questions involving price trends, comparisons between dates, highest/lowest prices in a period, etc.
    """
    try:
        if 'historical_df' not in st.session_state:
            return "Historical price data is not loaded. Please upload a CSV file first."
        
        df = st.session_state.historical_df.copy()
        df['Arrival_Date'] = pd.to_datetime(df['Arrival_Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Arrival_Date'])
        df = df.sort_values('Arrival_Date').reset_index(drop=True)
        df['Year'] = df['Arrival_Date'].dt.year
        df['Month'] = df['Arrival_Date'].dt.month

        api_key = st.session_state.get('perplexity_api_key', '')
        if not api_key:
            return "Perplexity API key not configured."
            
        llm = PerplexityLLM(api_key=api_key)

        from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
        pandas_agent = create_pandas_dataframe_agent(
            llm, df, verbose=True, allow_dangerous_code=True,
            handle_parsing_errors="Check your output and make sure it conforms!",
        )

        full_prompt = f"""
        Analyze the provided dataframe `df` to answer the user's question about historical prices.
        User's question: '{query}'
        Provide a direct, factual answer based on your analysis.
        """
        response = pandas_agent.invoke({"input": full_prompt})
        return response['output']

    except Exception as e:
        return f"An error occurred while analyzing historical prices: {e}"

# --- Tool 3: Government Scheme Retriever ---
def build_vector_store_from_uploaded_files(uploaded_files):
    """Builds vector store from uploaded PDF files."""
    try:
        all_docs = []
        temp_files = []
        
        for uploaded_file in uploaded_files:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
                temp_files.append(temp_file_path)
            
            # Load documents from the temporary file
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            all_docs.extend(docs)
        
        if not all_docs:
            return False, "No documents could be loaded from the uploaded files."
        
        # Split documents and create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Store in session state
        st.session_state.vector_store = vector_store
        st.session_state.vector_store_built = True
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)
        
        return True, f"Vector store built successfully from {len(uploaded_files)} files with {len(chunks)} chunks."
        
    except Exception as e:
        return False, f"Error building vector store: {e}"

@tool
def retrieve_scheme_information(query: str) -> str:
    """
    Searches a knowledge base of government agricultural schemes to find relevant information.
    Use for questions about financial aid, subsidies, insurance, etc.
    """
    try:
        if not st.session_state.get('vector_store_built', False):
            return "Vector store is not available. Please upload PDF files first."
        
        vector_store = st.session_state.vector_store
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        if not context:
            return "Could not find any specific schemes for your question."
        
        api_key = st.session_state.get('perplexity_api_key', '')
        if not api_key:
            return "Perplexity API key not configured."
            
        llm = PerplexityLLM(api_key=api_key)
        synthesis_prompt = f"Based on the following info, answer the user's question about schemes:\n\nContext:\n{context}\n\nUser's question: {query}\n\nAnswer:"
        return llm._call(synthesis_prompt)
        
    except Exception as e:
        return f"Error retrieving scheme info: {e}"

# --- Tool 4: Plant Identification and Inquiry ---
@tool
def identify_and_inquire_about_plant(query: str) -> str:
    """
    Identifies a plant from an uploaded image and answers a follow-up question.
    The query should mention the image name that was uploaded.
    """
    try:
        # Extract image name from query
        uploaded_images = st.session_state.get('uploaded_images', {})
        
        if not uploaded_images:
            return "No images have been uploaded. Please upload an image first."
        
        # Try to find which image the user is referring to
        image_path = None
        image_name = None
        
        # Look for image file extensions in the query
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for name, path in uploaded_images.items():
            # Check if the image name is mentioned in the query
            name_without_ext = name.lower().split('.')[0]
            if (name.lower() in query.lower() or 
                name_without_ext in query.lower() or 
                any(ext in query for ext in image_extensions) or
                len(uploaded_images) == 1):
                image_path = path
                image_name = name
                break
        
        if not image_path:
            available_images = ", ".join(uploaded_images.keys())
            return f"Could not determine which image you're referring to. Available images: {available_images}. Please mention the image name in your question."
        
        plantnet_api_key = st.session_state.get('plantnet_api_key', '')
        if not plantnet_api_key:
            return "PlantNet API key not configured. Please add it in the sidebar."
        
        # Identify the plant using PlantNet API
        identification_result = ""
        try:
            with open(image_path, 'rb') as image_file:
                req = requests.Request(
                    'POST', url="https://my-api.plantnet.org/v2/identify/all",
                    files={'images': image_file}, data={'organs': ['leaf', 'flower']},
                    params={'api-key': plantnet_api_key}
                )
                response = requests.Session().send(req.prepare())
                response.raise_for_status()
                json_result = response.json()

                if json_result.get('results'):
                    best_match = json_result['results'][0]
                    scientific_name = best_match['species']['scientificNameWithoutAuthor']
                    common_names = best_match['species'].get('commonNames', ['N/A'])
                    confidence = best_match.get('score', 0)
                    identification_result = f"Plant identified: {scientific_name}"
                    if common_names and common_names != ['N/A']:
                        identification_result += f" (Common names: {', '.join(common_names[:2])})"
                    identification_result += f" with {confidence:.1%} confidence."
                else:
                    return "Could not identify the plant from the image. The image may be unclear or the plant may not be in the database."
        except requests.RequestException as e:
            return f"Error calling PlantNet API: {e}"
        
        # Use LLM to answer the follow-up question
        api_key = st.session_state.get('perplexity_api_key', '')
        if not api_key:
            return f"{identification_result}\n\nPerplexity API key not configured for detailed analysis."
            
        llm = PerplexityLLM(api_key=api_key)
        follow_up_prompt = f"""Based on the plant identification: {identification_result}
        
        Answer this farmer's question: {query}
        
        Provide practical advice about:
        1. Whether this is a weed or beneficial plant
        2. How to manage/remove it if it's a weed
        3. Any other relevant farming advice
        
        Keep the response concise and practical."""
        
        follow_up_answer = llm._call(follow_up_prompt)

        return f"{identification_result}\n\n{follow_up_answer}"
        
    except Exception as e:
        return f"Error processing plant identification: {e}"

# --- Main RAG Agent ---
def create_main_agent():
    """Creates and returns the main RAG agent that orchestrates the tools."""
    tools = [
        get_latest_mandi_price,
        query_historical_prices,
        retrieve_scheme_information,
        identify_and_inquire_about_plant,
    ]

    template = """
    You are a helpful assistant for farmers in India. Your goal is to answer questions by combining information from specialized tools.
    Analyze the user's question and use one or more of the available tools to find the answer.
    You MUST provide the final answer in the same language as the original user's question.

    IMPORTANT: Keep your reasoning concise and direct. Don't repeat actions or get stuck in loops.

    TOOLS:
    ------
    You have access to the following tools:
    {tools}

    To use a tool, please use the following format:
    ```
    Thought: Do I need to use a tool? Yes
    Action: The action to take, should be one of [{tool_names}]
    Action Input: The input to the action
    Observation: The result of the action
    ```

    When you have enough information to answer, immediately provide your final answer:
    ```
    Thought: I now have enough information to answer the question.
    Final Answer: [Your comprehensive response in the same language as the user's question]
    ```

    Question: {input}
    {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template)
    api_key = st.session_state.get('perplexity_api_key', '')
    
    if not api_key:
        return None
        
    llm = PerplexityLLM(api_key=api_key)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5,  # Limit iterations to prevent infinite loops
        max_execution_time=60,  # 60 second timeout
        early_stopping_method="generate"  # Stop early if possible
    )

    return agent_executor

# --- Streamlit UI ---
def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Krishi Samadhan – Farmonomics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-powered farming companion for prices, schemes, and plant identification</p>', unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("🔑 API Keys")
        
        # Try to load from Streamlit secrets first, fallback to manual input
        try:
            perplexity_key = st.secrets.get("PERPLEXITY_API_KEY", "")
            plantnet_key = st.secrets.get("PLANTNET_API_KEY", "")
            
            if perplexity_key:
                st.session_state.perplexity_api_key = perplexity_key
                st.success("✅ Perplexity API key loaded from secrets")
            else:
                perplexity_key = st.text_input("Perplexity API Key", type="password", 
                                             value=st.session_state.get('perplexity_api_key', ''))
                if perplexity_key:
                    st.session_state.perplexity_api_key = perplexity_key
            
            if plantnet_key:
                st.session_state.plantnet_api_key = plantnet_key
                st.success("✅ PlantNet API key loaded from secrets")
            else:
                plantnet_key = st.text_input("PlantNet API Key", type="password",
                                           value=st.session_state.get('plantnet_api_key', ''))
                if plantnet_key:
                    st.session_state.plantnet_api_key = plantnet_key
                    
        except Exception:
            # Fallback to manual input if secrets are not available
            perplexity_key = st.text_input("Perplexity API Key", type="password", 
                                         value=st.session_state.get('perplexity_api_key', ''))
            if perplexity_key:
                st.session_state.perplexity_api_key = perplexity_key
            
            plantnet_key = st.text_input("PlantNet API Key", type="password",
                                       value=st.session_state.get('plantnet_api_key', ''))
            if plantnet_key:
                st.session_state.plantnet_api_key = plantnet_key
        
        st.divider()
        
        # File uploads
        st.subheader("📄 Upload Files")
        
        # Historical data upload
        st.write("**Historical Price Data (CSV)**")
        uploaded_csv = st.file_uploader("Upload CSV file", type=['csv'], key="csv_upload")
        if uploaded_csv is not None:
            try:
                df = pd.read_csv(uploaded_csv)
                st.session_state.historical_df = df
                st.session_state.historical_data_loaded = True
                st.success(f"✅ Loaded CSV with {len(df)} records")
                st.write("Preview:", df.head(2))
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
        
        # Scheme documents upload
        st.write("**Government Scheme Documents (PDF)**")
        uploaded_pdfs = st.file_uploader("Upload PDF files", type=['pdf'], 
                                       accept_multiple_files=True, key="pdf_upload")
        if uploaded_pdfs:
            if st.button("🔧 Build Knowledge Base"):
                with st.spinner("Building vector store..."):
                    success, message = build_vector_store_from_uploaded_files(uploaded_pdfs)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        # Image uploads for plant identification
        st.write("**Plant Images**")
        uploaded_images = st.file_uploader("Upload plant images", type=['jpg', 'jpeg', 'png'], 
                                         accept_multiple_files=True, key="image_upload")
        if uploaded_images:
            st.session_state.uploaded_images = {}
            for img in uploaded_images:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{img.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(img.getvalue())
                    st.session_state.uploaded_images[img.name] = tmp_file.name
            
            st.success(f"✅ Uploaded {len(uploaded_images)} images")
            for img_name in st.session_state.uploaded_images.keys():
                st.write(f"• {img_name}")

        st.divider()
        
        # Status indicators
        st.subheader("📊 System Status")
        api_status = "✅" if st.session_state.get('perplexity_api_key') else "❌"
        st.write(f"Perplexity API: {api_status}")
        
        plant_status = "✅" if st.session_state.get('plantnet_api_key') else "❌"
        st.write(f"PlantNet API: {plant_status}")
        
        csv_status = "✅" if st.session_state.get('historical_data_loaded') else "❌"
        st.write(f"Historical Data: {csv_status}")
        
        vector_status = "✅" if st.session_state.get('vector_store_built') else "❌"
        st.write(f"Knowledge Base: {vector_status}")
        
        images_status = "✅" if st.session_state.get('uploaded_images') else "❌"
        st.write(f"Plant Images: {images_status}")

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Your Question")
        
        # Tool descriptions
        with st.expander("🔧 Available Tools", expanded=False):
            st.markdown("""
            <div class="tool-card">
                <strong>🏪 Latest Mandi Prices</strong><br>
                Get current wheat prices from Kanpur market
            </div>
            <div class="tool-card">
                <strong>📈 Historical Price Analysis</strong><br>
                Analyze price trends and patterns from uploaded data
            </div>
            <div class="tool-card">
                <strong>📋 Government Schemes</strong><br>
                Find relevant agricultural schemes and subsidies
            </div>
            <div class="tool-card">
                <strong>🌱 Plant Identification</strong><br>
                Identify plants from images and get care advice
            </div>
            """, unsafe_allow_html=True)
        
        # Question input
        user_question = st.text_area("Enter your question here:", 
                                   placeholder="Example: What is the current wheat price in Kanpur?",
                                   height=100)
        
        # Query button
        if st.button("🚀 Get Answer", type="primary"):
            if not user_question.strip():
                st.warning("Please enter a question.")
            elif not st.session_state.get('perplexity_api_key'):
                st.error("Please configure your Perplexity API key in the sidebar.")
            else:
                # Create agent
                agent = create_main_agent()
                if agent is None:
                    st.error("Failed to create agent. Check your API configuration.")
                else:
                    with st.spinner("🤔 Thinking..."):
                        # Capture the agent's thinking process
                        with CaptureOutput() as captured:
                            start_time = time.time()
                            try:
                                result = agent.invoke({"input": user_question})
                                end_time = time.time()
                                
                                # Display results
                                st.success("✅ Answer generated successfully!")
                                
                                # Final answer
                                st.markdown("### 🎯 Final Answer")
                                st.markdown(result['output'])
                                
                                # Show metrics in col2
                                processing_time = end_time - start_time
                                st.session_state.last_processing_time = processing_time
                                st.session_state.last_thinking_process = captured.get_output()
                                
                            except Exception as e:
                                st.error(f"An error occurred: {e}")

    with col2:
        st.header("📊 Analysis")
        
        # Show thinking process
        if st.session_state.get('last_thinking_process'):
            with st.expander("🧠 Thinking Process", expanded=True):
                st.markdown(f"""
                <div class="thinking-process">
                <pre>{st.session_state.last_thinking_process}</pre>
                </div>
                """, unsafe_allow_html=True)
        
        # Show metrics
        if st.session_state.get('last_processing_time'):
            st.markdown("### ⚡ Performance Metrics")
            
            processing_time = st.session_state.last_processing_time
            
            st.markdown(f"""
            <div class="success-metric">
            <strong>Processing Time:</strong> {processing_time:.2f} seconds<br>
            <strong>Status:</strong> ✅ Success<br>
            <strong>Tools Used:</strong> {len([line for line in st.session_state.get('last_thinking_process', '').split('\n') if 'Action:' in line])}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence score based on processing time and API response
            if processing_time < 10:
                confidence = "High (Fast response)"
                color = "#4caf50"
            elif processing_time < 20:
                confidence = "Medium (Normal response)"
                color = "#ff9800"
            else:
                confidence = "Low (Slow response)"
                color = "#f44336"
            
            st.markdown(f"""
            <div class="confidence-score">
            <strong>Confidence Score:</strong> <span style="color: {color};">{confidence}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # API usage metrics
        if st.session_state.get('api_metrics'):
            st.markdown("### 📈 API Usage")
            metrics = st.session_state.api_metrics[-5:]  # Last 5 calls
            avg_time = sum(m['response_time'] for m in metrics) / len(metrics)
            total_calls = len(st.session_state.api_metrics)
            
            st.metric("Average Response Time", f"{avg_time:.2f}s")
            st.metric("Total API Calls", total_calls)

if __name__ == "__main__":
    main()
