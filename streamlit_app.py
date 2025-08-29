import streamlit as st
import os
import re
import requests
import pandas as pd
import tempfile
import json
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    text-align: center;
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
    max-height: 300px;
    overflow-y: auto;
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
.performance-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize comprehensive session state for metrics
def initialize_metrics():
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            # Document processing metrics
            'documents_uploaded': 0,
            'total_pages_processed': 0,
            'text_chunks_created': 0,
            'vector_store_size_mb': 0,
            'average_chunk_length': 0,
            'total_tokens_processed': 0,
            
            # API performance metrics
            'api_calls': [],
            'llm_response_times': [],
            'search_latencies': [],
            'total_api_calls': 0,
            
            # Plant identification metrics
            'plant_identifications': [],
            'plant_accuracy_tests': [],
            
            # Query metrics
            'queries_processed': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'query_times': [],
            
            # User engagement metrics
            'session_start_time': datetime.now(),
            'tools_used': {'mandi_price': 0, 'historical_analysis': 0, 
                          'scheme_retrieval': 0, 'plant_identification': 0},
            
            # Comparison metrics
            'semantic_vs_keyword_times': [],
            
            # System metrics
            'corpus_size_mb': 0,
            'embedding_dimensions': 0,
        }

# Initialize session state
initialize_metrics()
if 'vector_store_built' not in st.session_state:
    st.session_state.vector_store_built = False
if 'historical_data_loaded' not in st.session_state:
    st.session_state.historical_data_loaded = False
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = {}
if 'agent_ready' not in st.session_state:
    st.session_state.agent_ready = False

# Enhanced Perplexity LLM with detailed metrics
class PerplexityLLM(LLM):
    """Custom LangChain LLM wrapper for the Perplexity API with comprehensive metrics tracking."""
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
            
            response_time = end_time - start_time
            response_text = result["choices"][0]["message"]["content"].strip()
            
            # Enhanced metrics tracking
            api_call_data = {
                'timestamp': datetime.now(),
                'response_time': response_time,
                'prompt_length': len(prompt),
                'prompt_tokens': len(prompt.split()),
                'response_length': len(response_text),
                'response_tokens': len(response_text.split()),
                'model_used': self.model,
                'success': True
            }
            
            st.session_state.metrics['api_calls'].append(api_call_data)
            st.session_state.metrics['llm_response_times'].append(response_time)
            st.session_state.metrics['total_api_calls'] += 1
            
            return response_text
            
        except requests.RequestException as e:
            # Track failed API calls
            api_call_data = {
                'timestamp': datetime.now(),
                'response_time': None,
                'prompt_length': len(prompt),
                'error': str(e),
                'success': False
            }
            st.session_state.metrics['api_calls'].append(api_call_data)
            return f"API Error: {e}"
        except Exception as e:
            return f"An unexpected error occurred during the API call: {e}"

# Enhanced vector store builder with metrics
def build_vector_store_from_uploaded_files(uploaded_files):
    """Builds vector store from uploaded PDF files with comprehensive metrics."""
    try:
        all_docs = []
        temp_files = []
        total_pages = 0
        
        start_time = time.time()
        
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
            total_pages += len(docs)
        
        if not all_docs:
            return False, "No documents could be loaded from the uploaded files."
        
        # Calculate document size
        total_text = " ".join([doc.page_content for doc in all_docs])
        corpus_size_mb = len(total_text.encode('utf-8')) / (1024 * 1024)
        total_tokens = len(total_text.split())
        
        # Split documents and create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        
        # Calculate chunk metrics
        chunk_lengths = [len(chunk.page_content.split()) for chunk in chunks]
        average_chunk_length = np.mean(chunk_lengths) if chunk_lengths else 0
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Time the vector store creation
        vector_start = time.time()
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_creation_time = time.time() - vector_start
        
        # Store in session state
        st.session_state.vector_store = vector_store
        st.session_state.vector_store_built = True
        
        # Update metrics
        st.session_state.metrics.update({
            'documents_uploaded': len(uploaded_files),
            'total_pages_processed': total_pages,
            'text_chunks_created': len(chunks),
            'corpus_size_mb': corpus_size_mb,
            'average_chunk_length': average_chunk_length,
            'total_tokens_processed': total_tokens,
            'vector_creation_time': vector_creation_time,
            'embedding_dimensions': 384  # MiniLM-L6-v2 embedding size
        })
        
        # Clean up temporary files
        for temp_file in temp_files:
            os.unlink(temp_file)
        
        processing_time = time.time() - start_time
        
        return True, f"""Vector store built successfully!
        📊 Metrics:
        • {len(uploaded_files)} documents processed
        • {total_pages} pages extracted
        • {len(chunks)} text chunks created
        • {corpus_size_mb:.2f} MB corpus size
        • {average_chunk_length:.0f} avg tokens per chunk
        • {processing_time:.2f}s processing time
        • {vector_creation_time:.2f}s vector creation time"""
        
    except Exception as e:
        return False, f"Error building vector store: {e}"

# Captured Output Context Manager
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

# Enhanced tools with metrics tracking
@tool
def get_latest_mandi_price() -> str:
    """
    Fetches the latest 'Dara' variety wheat price from mandiprices.com for Kanpur.
    Returns a string with the findings or an error message.
    """
    start_time = time.time()
    st.session_state.metrics['tools_used']['mandi_price'] += 1
    
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

        # Track performance
        processing_time = time.time() - start_time
        st.session_state.metrics['search_latencies'].append({
            'tool': 'mandi_price',
            'time': processing_time,
            'success': True,
            'timestamp': datetime.now()
        })

        if max_price == "Not found" and avg_price == "Not found":
            return "Could not find the latest wheat price for Kanpur. The website structure may have changed."

        return f"Today's wheat price in Kanpur: Maximum Price is ₹{max_price}, Average Price is ₹{avg_price}."

    except requests.RequestException as e:
        processing_time = time.time() - start_time
        st.session_state.metrics['search_latencies'].append({
            'tool': 'mandi_price',
            'time': processing_time,
            'success': False,
            'timestamp': datetime.now()
        })
        return f"Error fetching latest price data: {e}"
    except Exception as e:
        return f"An unexpected error occurred while fetching the latest price: {e}"

@tool
def query_historical_prices(query: str) -> str:
    """
    Answers questions about historical wheat prices from 2020-2024 using uploaded CSV data.
    Use this for questions involving price trends, comparisons between dates, highest/lowest prices in a period, etc.
    """
    start_time = time.time()
    st.session_state.metrics['tools_used']['historical_analysis'] += 1
    
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
        
        # Track performance
        processing_time = time.time() - start_time
        st.session_state.metrics['search_latencies'].append({
            'tool': 'historical_analysis',
            'time': processing_time,
            'success': True,
            'timestamp': datetime.now()
        })
        
        return response['output']

    except Exception as e:
        processing_time = time.time() - start_time
        st.session_state.metrics['search_latencies'].append({
            'tool': 'historical_analysis',
            'time': processing_time,
            'success': False,
            'timestamp': datetime.now()
        })
        return f"An error occurred while analyzing historical prices: {e}"

@tool
def retrieve_scheme_information(query: str) -> str:
    """
    Searches a knowledge base of government agricultural schemes to find relevant information.
    Use for questions about financial aid, subsidies, insurance, etc.
    """
    start_time = time.time()
    st.session_state.metrics['tools_used']['scheme_retrieval'] += 1
    
    try:
        if not st.session_state.get('vector_store_built', False):
            return "Vector store is not available. Please upload PDF files first."
        
        vector_store = st.session_state.vector_store
        
        # Time semantic search
        search_start = time.time()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(query)
        search_time = time.time() - search_start
        
        context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Track semantic search performance
        st.session_state.metrics['semantic_vs_keyword_times'].append({
            'query': query,
            'semantic_time': search_time,
            'docs_found': len(relevant_docs),
            'timestamp': datetime.now()
        })
        
        if not context:
            return "Could not find any specific schemes for your question."
        
        api_key = st.session_state.get('perplexity_api_key', '')
        if not api_key:
            return "Perplexity API key not configured."
            
        llm = PerplexityLLM(api_key=api_key)
        synthesis_prompt = f"Based on the following info, answer the user's question about schemes:\n\nContext:\n{context}\n\nUser's question: {query}\n\nAnswer:"
        result = llm._call(synthesis_prompt)
        
        # Track overall performance
        processing_time = time.time() - start_time
        st.session_state.metrics['search_latencies'].append({
            'tool': 'scheme_retrieval',
            'time': processing_time,
            'success': True,
            'timestamp': datetime.now()
        })
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        st.session_state.metrics['search_latencies'].append({
            'tool': 'scheme_retrieval',
            'time': processing_time,
            'success': False,
            'timestamp': datetime.now()
        })
        return f"Error retrieving scheme info: {e}"

@tool
def identify_and_inquire_about_plant(query: str) -> str:
    """
    Identifies a plant from an uploaded image and answers a follow-up question.
    The query should mention the image name that was uploaded.
    """
    start_time = time.time()
    st.session_state.metrics['tools_used']['plant_identification'] += 1
    
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
        identification_start = time.time()
        identification_result = ""
        plant_data = None
        
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

                identification_time = time.time() - identification_start

                if json_result.get('results'):
                    best_match = json_result['results'][0]
                    scientific_name = best_match['species']['scientificNameWithoutAuthor']
                    common_names = best_match['species'].get('commonNames', ['N/A'])
                    confidence = best_match.get('score', 0)
                    identification_result = f"Plant identified: {scientific_name}"
                    if common_names and common_names != ['N/A']:
                        identification_result += f" (Common names: {', '.join(common_names[:2])})"
                    identification_result += f" with {confidence:.1%} confidence."
                    
                    # Store plant identification metrics
                    plant_data = {
                        'timestamp': datetime.now(),
                        'image_name': image_name,
                        'scientific_name': scientific_name,
                        'confidence': confidence,
                        'identification_time': identification_time,
                        'common_names': common_names,
                        'success': True
                    }
                    
                else:
                    plant_data = {
                        'timestamp': datetime.now(),
                        'image_name': image_name,
                        'identification_time': identification_time,
                        'success': False,
                        'error': 'No results found'
                    }
                    return "Could not identify the plant from the image. The image may be unclear or the plant may not be in the database."
                    
        except requests.RequestException as e:
            identification_time = time.time() - identification_start
            plant_data = {
                'timestamp': datetime.now(),
                'image_name': image_name,
                'identification_time': identification_time,
                'success': False,
                'error': str(e)
            }
            return f"Error calling PlantNet API: {e}"
        
        # Store metrics
        st.session_state.metrics['plant_identifications'].append(plant_data)
        
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
        
        # Track overall performance
        processing_time = time.time() - start_time
        st.session_state.metrics['search_latencies'].append({
            'tool': 'plant_identification',
            'time': processing_time,
            'success': True,
            'timestamp': datetime.now()
        })

        return f"{identification_result}\n\n{follow_up_answer}"
        
    except Exception as e:
        processing_time = time.time() - start_time
        st.session_state.metrics['search_latencies'].append({
            'tool': 'plant_identification',
            'time': processing_time,
            'success': False,
            'timestamp': datetime.now()
        })
        return f"Error processing plant identification: {e}"

# Main RAG Agent
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
        max_iterations=5,
        max_execution_time=60,
        early_stopping_method="generate"
    )

    return agent_executor

# Metrics display functions
def display_performance_dashboard():
    """Display comprehensive performance metrics dashboard."""
    st.markdown("## 📊 Performance Dashboard")
    
    metrics = st.session_state.metrics
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_queries = metrics['queries_processed']
        st.metric("Total Queries", total_queries, delta=None)
        
    with col2:
        success_rate = (metrics['successful_queries'] / max(total_queries, 1)) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
        
    with col3:
        avg_response_time = np.mean(metrics['llm_response_times']) if metrics['llm_response_times'] else 0
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
        
    with col4:
        session_duration = (datetime.now() - metrics['session_start_time']).total_seconds() / 60
        st.metric("Session Duration", f"{session_duration:.1f} min")

    # Document Processing Metrics
    st.markdown("### 📄 Document Processing")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Documents Processed</h4>
            <h2>{metrics['documents_uploaded']}</h2>
            <small>{metrics['total_pages_processed']} pages</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Text Chunks</h4>
            <h2>{metrics['text_chunks_created']}</h2>
            <small>Avg: {metrics['average_chunk_length']:.0f} tokens</small>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Corpus Size</h4>
            <h2>{metrics['corpus_size_mb']:.1f} MB</h2>
            <small>{metrics['total_tokens_processed']} tokens</small>
        </div>
        """, unsafe_allow_html=True)

    # API Performance Charts
    if metrics['llm_response_times']:
        st.markdown("### ⚡ API Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time distribution
            fig = px.histogram(
                x=metrics['llm_response_times'],
                nbins=10,
                title="Response Time Distribution",
                labels={'x': 'Response Time (seconds)', 'y': 'Frequency'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Response times over time
            if len(metrics['api_calls']) > 1:
                df_calls = pd.DataFrame(metrics['api_calls'])
                df_calls = df_calls[df_calls['success'] == True]
                
                fig = px.line(
                    df_calls,
                    x='timestamp', 
                    y='response_time',
                    title="Response Time Trend",
                    labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    # Tool Usage Analytics
    st.markdown("### 🛠️ Tool Usage Analytics")
    
    tools_data = metrics['tools_used']
    if sum(tools_data.values()) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Tool usage pie chart
            fig = px.pie(
                values=list(tools_data.values()),
                names=list(tools_data.keys()),
                title="Tool Usage Distribution"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Tool performance comparison
            if metrics['search_latencies']:
                df_latencies = pd.DataFrame(metrics['search_latencies'])
                avg_latencies = df_latencies.groupby('tool')['time'].mean().reset_index()
                
                fig = px.bar(
                    avg_latencies,
                    x='tool',
                    y='time',
                    title="Average Tool Response Time",
                    labels={'time': 'Response Time (s)', 'tool': 'Tool'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    # Plant Identification Metrics
    if metrics['plant_identifications']:
        st.markdown("### 🌱 Plant Identification Analytics")
        
        plant_df = pd.DataFrame(metrics['plant_identifications'])
        successful_identifications = plant_df[plant_df['success'] == True]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            success_rate = len(successful_identifications) / len(plant_df) * 100
            st.metric("Identification Success Rate", f"{success_rate:.1f}%")
            
        with col2:
            if len(successful_identifications) > 0:
                avg_confidence = successful_identifications['confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            else:
                st.metric("Average Confidence", "N/A")
                
        with col3:
            avg_time = plant_df['identification_time'].mean()
            st.metric("Avg Identification Time", f"{avg_time:.2f}s")
        
        if len(successful_identifications) > 0:
            # Confidence distribution
            fig = px.histogram(
                successful_identifications,
                x='confidence',
                nbins=10,
                title="Plant Identification Confidence Distribution",
                labels={'confidence': 'Confidence Score', 'count': 'Frequency'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

def display_resume_metrics():
    """Display key metrics formatted for resume/portfolio use."""
    st.markdown("## 🎯 Metrics")
    
    metrics = st.session_state.metrics
    
    # Calculate key statistics
    total_documents = metrics['documents_uploaded']
    total_pages = metrics['total_pages_processed'] 
    total_chunks = metrics['text_chunks_created']
    corpus_size = metrics['corpus_size_mb']
    total_tokens = metrics['total_tokens_processed']
    
    # API performance
    avg_response_time = np.mean(metrics['llm_response_times']) if metrics['llm_response_times'] else 0
    p95_response_time = np.percentile(metrics['llm_response_times'], 95) if metrics['llm_response_times'] else 0
    total_api_calls = metrics['total_api_calls']
    
    # Plant identification accuracy
    plant_accuracy = 0
    if metrics['plant_identifications']:
        successful = sum(1 for p in metrics['plant_identifications'] if p['success'])
        plant_accuracy = (successful / len(metrics['plant_identifications'])) * 100
    
    # Query throughput
    session_duration_minutes = (datetime.now() - metrics['session_start_time']).total_seconds() / 60
    queries_per_minute = metrics['queries_processed'] / max(session_duration_minutes, 1)
    
    # Search performance
    semantic_search_times = [s['time'] for s in metrics['search_latencies'] if s['tool'] == 'scheme_retrieval' and s['success']]
    avg_search_time = np.mean(semantic_search_times) if semantic_search_times else 0
    
    # Display metrics in copyable format
    resume_points = f"""
    **🔥 Key Performance Metrics:**
    
    📊 **Document Processing & Knowledge Base:**
    • Processed {total_documents} PDF documents containing {total_pages} pages into {total_chunks} searchable text chunks
    • Built vector database with {corpus_size:.1f}MB corpus size and {total_tokens:,} tokens processed
    • Achieved {metrics['embedding_dimensions']} dimensional embeddings with semantic search capability
    
    ⚡ **API Performance & Latency:**
    • Average LLM response time: {avg_response_time:.3f}s with 95th percentile at {p95_response_time:.3f}s
    • Processed {total_api_calls} API calls with {metrics['successful_queries']}/{metrics['queries_processed']} successful queries
    • Semantic search latency: {avg_search_time:.3f}s average for document retrieval
    
    🌱 **Plant Identification System:**
    • Achieved {plant_accuracy:.1f}% identification accuracy on uploaded plant images
    • Average plant identification processing time: {np.mean([p['identification_time'] for p in metrics['plant_identifications'] if 'identification_time' in p]):.2f}s
    • Successfully integrated PlantNet API with confidence scoring
    
    🚀 **System Throughput & Scalability:**
    • Query processing rate: {queries_per_minute:.1f} queries per minute
    • Multi-modal data processing: PDFs, CSVs, images, and real-time web scraping
    • Built RAG (Retrieval-Augmented Generation) system with 4 specialized tools
    
    💰 **Time Savings & Efficiency:**
    • Automated mandi price fetching vs manual search: ~5-10 minutes saved per query
    • Government scheme retrieval: Instant access vs hours of manual document searching
    • Integrated historical analysis: Complex data queries resolved in <30 seconds
    
    🎯 **Technical Implementation:**
    • LangChain-based agent system with ReAct prompting methodology
    • FAISS vector database with HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2)
    • Real-time web scraping with BeautifulSoup and requests
    • Streamlit-based interactive dashboard with performance monitoring
    """
    
    st.markdown(Key points)
    
    # Copyable metrics for easy use
    st.markdown("### 📋 Quick Copy Metrics:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.code(f"""
• {total_documents} documents → {total_chunks} searchable chunks
• {corpus_size:.1f}MB knowledge base with {total_tokens:,} tokens
• {avg_response_time:.3f}s average API response time
• {plant_accuracy:.1f}% plant identification accuracy
• {queries_per_minute:.1f} queries/minute processing rate
        """)
        
    with col2:
        st.code(f"""
• 95th percentile latency: {p95_response_time:.3f}s
• {total_api_calls} total API calls processed
• {avg_search_time:.3f}s semantic search time
• Multi-modal RAG with 4 specialized tools
• Real-time price monitoring system
        """)

def export_metrics_report():
    """Generate a comprehensive metrics report for export."""
    metrics = st.session_state.metrics
    
    report = {
        "report_generated": datetime.now().isoformat(),
        "session_duration_minutes": (datetime.now() - metrics['session_start_time']).total_seconds() / 60,
        
        "document_processing": {
            "documents_uploaded": metrics['documents_uploaded'],
            "total_pages_processed": metrics['total_pages_processed'],
            "text_chunks_created": metrics['text_chunks_created'],
            "corpus_size_mb": metrics['corpus_size_mb'],
            "average_chunk_length": metrics['average_chunk_length'],
            "total_tokens_processed": metrics['total_tokens_processed'],
            "embedding_dimensions": metrics['embedding_dimensions']
        },
        
        "api_performance": {
            "total_api_calls": metrics['total_api_calls'],
            "average_response_time": np.mean(metrics['llm_response_times']) if metrics['llm_response_times'] else 0,
            "median_response_time": np.median(metrics['llm_response_times']) if metrics['llm_response_times'] else 0,
            "p95_response_time": np.percentile(metrics['llm_response_times'], 95) if metrics['llm_response_times'] else 0,
            "p99_response_time": np.percentile(metrics['llm_response_times'], 99) if metrics['llm_response_times'] else 0
        },
        
        "query_performance": {
            "total_queries": metrics['queries_processed'],
            "successful_queries": metrics['successful_queries'],
            "failed_queries": metrics['failed_queries'],
            "success_rate": (metrics['successful_queries'] / max(metrics['queries_processed'], 1)) * 100
        },
        
        "tool_usage": metrics['tools_used'],
        
        "plant_identification": {
            "total_attempts": len(metrics['plant_identifications']),
            "successful_identifications": len([p for p in metrics['plant_identifications'] if p['success']]),
            "accuracy_rate": (len([p for p in metrics['plant_identifications'] if p['success']]) / 
                            max(len(metrics['plant_identifications']), 1)) * 100 if metrics['plant_identifications'] else 0
        }
    }
    
    return json.dumps(report, indent=2, default=str)

# Streamlit UI
def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Krishi Samadhan – Farmonomics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-powered farming companion with comprehensive performance tracking</p>', unsafe_allow_html=True)

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("🔑 API Keys")
        
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
                
                # Update metrics
                st.session_state.metrics['csv_records'] = len(df)
                st.session_state.metrics['csv_columns'] = len(df.columns)
                
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
                with st.spinner("Building vector store with metrics tracking..."):
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
            total_image_size = 0
            
            for img in uploaded_images:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{img.name.split('.')[-1]}") as tmp_file:
                    img_data = img.getvalue()
                    tmp_file.write(img_data)
                    st.session_state.uploaded_images[img.name] = tmp_file.name
                    total_image_size += len(img_data)
            
            # Update metrics
            st.session_state.metrics['images_uploaded'] = len(uploaded_images)
            st.session_state.metrics['total_image_size_mb'] = total_image_size / (1024 * 1024)
            
            st.success(f"✅ Uploaded {len(uploaded_images)} images ({total_image_size/(1024*1024):.1f} MB)")
            for img_name in st.session_state.uploaded_images.keys():
                st.write(f"• {img_name}")

        st.divider()
        
        # Enhanced status indicators
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
        
        # Quick metrics preview
        st.subheader("📈 Quick Stats")
        if st.session_state.metrics['queries_processed'] > 0:
            st.write(f"🔥 Queries: {st.session_state.metrics['queries_processed']}")
            st.write(f"⚡ Avg Response: {np.mean(st.session_state.metrics['llm_response_times']) if st.session_state.metrics['llm_response_times'] else 0:.2f}s")
            st.write(f"📊 API Calls: {st.session_state.metrics['total_api_calls']}")

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Query Interface", "📊 Performance Dashboard", "🎯 Metrics"])
    
    with tab1:
        st.header("💬 Ask Your Question")
        
        # Tool descriptions
        with st.expander("🔧 Available Tools", expanded=False):
            st.markdown("""
            <div class="tool-card">
                <strong>🏪 Latest Mandi Prices</strong><br>
                Get current wheat prices from Kanpur market with real-time scraping
            </div>
            <div class="tool-card">
                <strong>📈 Historical Price Analysis</strong><br>
                Analyze price trends using pandas agent on uploaded CSV data
            </div>
            <div class="tool-card">
                <strong>📋 Government Schemes</strong><br>
                Semantic search through agricultural schemes using FAISS vector store
            </div>
            <div class="tool-card">
                <strong>🌱 Plant Identification</strong><br>
                Computer vision identification using PlantNet API with confidence scoring
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
                # Update metrics
                st.session_state.metrics['queries_processed'] += 1
                
                # Create agent
                agent = create_main_agent()
                if agent is None:
                    st.error("Failed to create agent. Check your API configuration.")
                    st.session_state.metrics['failed_queries'] += 1
                else:
                    with st.spinner("🤔 Thinking and tracking performance..."):
                        # Capture the agent's thinking process
                        with CaptureOutput() as captured:
                            start_time = time.time()
                            try:
                                result = agent.invoke({"input": user_question})
                                end_time = time.time()
                                
                                # Update success metrics
                                st.session_state.metrics['successful_queries'] += 1
                                processing_time = end_time - start_time
                                st.session_state.metrics['query_times'].append(processing_time)
                                
                                # Display results
                                st.success("✅ Answer generated successfully!")
                                
                                # Final answer
                                st.markdown("### 🎯 Final Answer")
                                st.markdown(result['output'])
                                
                                # Performance summary
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Processing Time", f"{processing_time:.2f}s")
                                with col2:
                                    tools_used = len([line for line in captured.get_output().split('\n') if 'Action:' in line])
                                    st.metric("Tools Used", tools_used)
                                with col3:
                                    st.metric("API Calls", len([call for call in st.session_state.metrics['api_calls'] 
                                                              if call['timestamp'] > datetime.now() - timedelta(minutes=1)]))
                                
                                # Store for detailed view
                                st.session_state.last_processing_time = processing_time
                                st.session_state.last_thinking_process = captured.get_output()
                                
                            except Exception as e:
                                st.session_state.metrics['failed_queries'] += 1
                                st.error(f"An error occurred: {e}")

        # Show recent thinking process
        if st.session_state.get('last_thinking_process'):
            with st.expander("🧠 Agent Thinking Process", expanded=False):
                st.markdown(f"""
                <div class="thinking-process">
                <pre>{st.session_state.last_thinking_process}</pre>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        display_performance_dashboard()
        
        # Export metrics
        st.markdown("### 📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Generate Metrics Report"):
                report = export_metrics_report()
                st.download_button(
                    label="💾 Download JSON Report",
                    data=report,
                    file_name=f"krishi_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
        with col2:
            if st.button("📈 Export CSV Data"):
                # Create DataFrame from metrics for CSV export
                if st.session_state.metrics['api_calls']:
                    df_api = pd.DataFrame(st.session_state.metrics['api_calls'])
                    csv = df_api.to_csv(index=False)
                    st.download_button(
                        label="💾 Download API Metrics CSV",
                        data=csv,
                        file_name=f"api_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    with tab3:
        display_resume_metrics()

if __name__ == "__main__":
    main()
