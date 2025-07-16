# Simple RAG App

A basic RAG (Retrieval-Augmented Generation) app that answers questions about capital cities.

## What it does
- Loads city data from a JSON file
- Finds relevant cities when you ask a question
- Uses Llama AI to give you an answer

## Setup
1. Install packages: `pip install -r requirements.txt`
2. Put your API key in `.env` file: `META_LLAMA_API_KEY=your_key_here`
3. Run: `python rag_app.py`

## How to use
- Type a question about cities (like "Tell me about Tokyo")
- Get an AI answer based on the city data
- Type 'q' to quit

## Files
- `rag_app.py` - Main app
- `capital_cities.json` - City data
- `requirements.txt` - Packages needed
