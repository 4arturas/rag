# Custom RAG AI Agent

This project implements a Retrieval-Augmented Generation (RAG) AI agent following the tutorial from https://deno.com/blog/build-custom-rag-ai-agent. The agent uses Deno, LangChain, and Ollama to create a local RAG system that can answer questions based on custom data sources.

## Technologies Used

- **Node.js**: JavaScript runtime for the application
- **LangChain**: AI framework for creating AI workflows
- **LangGraph**: For creating stateful AI agents
- **Ollama**: Local LLM server for running models

## Models Used

- **Llama 3.2 (3 billion parameters)**: Supports tooling for calling additional functions (`llama3.2`)
- **Mixedbread's embedding model**: `mxbai-embed-large` - transforms text into searchable format

## Prerequisites

1. Node.js installed on your system
2. Ollama installed (https://ollama.com/)

## Setup

1. Install required dependencies:
   ```bash
   npm install
   ```

2. Pull required models:
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large
   ```

3. Make sure Ollama is running in a separate terminal before running the application:
   ```bash
   ollama serve
   ```

## Checking Ollama

You can check if Ollama is running and has the required models with:
```bash
node check-ollama.js
```

## Running the Application

```bash
npm start
```
```