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

## Why Multiple Models?

This RAG system uses different models for different tasks rather than a single model for the following reasons:

### 1. **Specialized Capabilities**
Different models excel at different tasks:
- **Embedding models** (like mxbai-embed-large): Optimized for creating vector representations of text, crucial for similarity search
- **Generation models** (like deepseek-r1:8b): Better at producing human-readable, coherent responses
- **Agent models** (like llama3.2:3b): Efficient for decision-making and tool selection

### 2. **Resource Efficiency**
- **Embedding models** are specifically designed for creating semantic representations efficiently
- **Smaller models** work well for tool selection decisions, saving computational resources
- **Larger models** are reserved for tasks requiring higher complexity, like final response generation

### 3. **Performance Optimization**
- **Specialized models** perform their designated tasks better than a general-purpose model
- **Embedding models** create more accurate similarity matches for retrieval
- **Generation models** produce higher-quality, more coherent final answers

### 4. **Cost Management**
- Running a large model for every decision (like whether to retrieve documents) would be expensive
- Smaller, specialized models handle routine tasks more efficiently
- Large models are used only when their full capacity is needed

### 5. **Quality Control**
- Different models can be tuned specifically for their role
- For example, an agent model optimized for function calling makes better tool selection decisions
- A generation model fine-tuned for responding to queries produces better final outputs

In this RAG system, this means:
- The **agent model** decides whether to retrieve information (a classification/task selection task)
- The **embedding model** creates the vector representations for similarity search
- The **generation model** crafts the final response using retrieved context

This approach results in a more efficient, higher-quality system than using one model for everything, which would likely be either too resource-intensive or not optimized for specific tasks.