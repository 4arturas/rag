# Custom RAG AI Agent

This project implements a Retrieval-Augmented Generation (RAG) AI agent using Node.js, LangChain, LangGraph, and Ollama.

## Implementation Overview

The `index.js` file implements a RAG workflow with the following key logic:

- **Document Loading**: Scrapes content from Deno blog posts using CheerioWebBaseLoader
- **Text Splitting**: Splits documents into chunks using RecursiveCharacterTextSplitter
- **Vector Storage**: Creates embeddings using OllamaEmbeddings and stores them in HNSWLib vector store
- **Graph Workflow**: Implements a stateful graph with nodes for:
  - Agent (decides whether to retrieve documents)
  - Retrieval (searches for relevant documents)
  - Relevance Assessment (grades document relevance)
  - Query Transformation (rewrites query if documents aren't relevant)
  - Generation (creates final response based on context)

The system uses a conditional workflow that checks document relevance and loops back to transform the query if documents aren't sufficiently relevant.