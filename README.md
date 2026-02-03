# Telegram RAG Chatbot (Azerbaijan Labour Code)

This repository demonstrates how to build a **Telegram chatbot** powered by **Retrieval-Augmented Generation (RAG)**.  
The bot is designed to answer questions related to the **Azerbaijan Labour Code** by retrieving relevant text chunks and generating grounded responses.

## What’s inside
- Telegram bot integration
- RAG pipeline (chunking → embeddings → retrieval → LLM answer)
- Local FAISS Vector DB

## Requirements
- Python 3.9+ recommended
- A Telegram Bot Token (from BotFather)
- An OpenAI API key (or your configured LLM provider key)

Install dependencies:
```bash
pip install -r requirements.txt
