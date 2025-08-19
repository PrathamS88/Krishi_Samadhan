# Krishi_Samadhan
# 🌾 Farmer Assistant RAG Agent

An AI-powered farming companion that provides:
- Real-time wheat prices from Kanpur market
- Historical price analysis and trends
- Government agricultural scheme information
- Plant identification and care advice

## Features

- **4 Specialized Tools**: Market prices, historical analysis, scheme lookup, plant ID
- **File Upload Support**: CSV data, PDF documents, plant images
- **Thinking Process Visualization**: See how the AI makes decisions
- **Performance Metrics**: Response times and confidence scores
- **Multi-language Support**: Responds in the same language as your question

## Setup

1. Get API keys:
   - [Perplexity AI API](https://perplexity.ai/) - for LLM processing
   - [PlantNet API](https://my.plantnet.org/) - for plant identification

2. Upload your files:
   - Historical price CSV data
   - Government scheme PDFs
   - Plant images for identification

3. Start asking questions!

## Usage Examples

- "What is today's wheat price in Kanpur?"
- "Show me price trends for the last 6 months"
- "What schemes are available for seed purchase?"
- "Identify this plant in my uploaded image"

## Technology Stack

- **Frontend**: Streamlit
- **LLM**: Perplexity AI
- **Embeddings**: HuggingFace Sentence Transformers
- **Vector Store**: FAISS
- **Plant ID**: PlantNet API
- **Web Scraping**: BeautifulSoup

## Deployment

This app is designed to run on Streamlit Community Cloud. Simply connect your GitHub repository and deploy!

---
Made with ❤️ for farmers
