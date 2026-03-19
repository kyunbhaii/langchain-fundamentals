# LangChain Fundamentals

A hands-on guide to LangChain covering Models, Prompts, Chains, Runnables, RAG pipelines, and Tools with practical Python examples.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## Topics Covered

| # | Module | What You'll Learn |
|---|--------|-------------------|
| 1 | [LangChain Models](#1-langchain-models) | LLMs, Chat Models, Embedding Models |
| 2 | [Prompts](#2-langchain-prompts) | Static/Dynamic Prompts, Chat Templates, Message Placeholders |
| 3 | [Structured Output](#3-structured-output) | TypedDict, Pydantic, JSON Parsers, Output Parsers |
| 4 | [Chains](#4-langchain-chains) | Simple, Sequential, Parallel, Conditional Chains |
| 5 | [Runnables (LCEL)](#5-langchain-runnables-lcel) | RunnableSequence, Parallel, Passthrough, Lambda, Branch |
| 6 | [RAG](#6-rag-retrieval-augmented-generation) | Document Loaders, Text Splitters, Vector DBs, Retrievers |
| 7 | [Tools](#7-tools) | Tool Creation, Tool Calling, Custom Tools |

---

## Project Structure

```
langchain-fundamentals/
│
├── 1_LangChain_Models/
│   ├── 1. LLMs/
│   ├── 2. ChatModels/
│   └── 3. EmbeddingModels/
│
├── 2_LangChain_Prompts/
│   ├── Static & Dynamic prompts
│   ├── Chatbot with memory
│   ├── Chat Prompt Templates
│   └── Message Placeholders
│
├── 3_LangChain_StructureOP/
│   ├── with_structured_output (TypedDict, Pydantic, JSON)
│   └── Output Parsers (Str, JSON, Pydantic)
│
├── 4_LangChain_Chains/
│   ├── Simple & Sequential Chains
│   ├── Parallel Chains
│   └── Conditional Chains
│
├── 5_LangChain_Runnables/
│   ├── RunnableSequence, Parallel, Passthrough
│   ├── RunnableLambda, RunnableBranch
│   └── LCEL (LangChain Expression Language)
│
├── 6_RAG/
│   ├── 1_DocumentLoaders/
│   ├── 2_TextSplitters/
│   ├── 3_VectorDB/
│   ├── 4_Retrievers/
│   └── YT_Chatbot.ipynb
│
└── 7_Tools/
    ├── Tool creation & calling
    └── Currency Conversion Tool (custom tool example)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- API keys for the models you want to use (OpenAI, Google Gemini, HuggingFace)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kyunbhaii/langchain-fundamentals.git
   cd langchain-fundamentals
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install langchain langchain-openai langchain-google-genai langchain-huggingface
   pip install langchain-community chromadb pypdf sentence-transformers
   pip install streamlit python-dotenv
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
   ```

---

## Module Breakdown

### 1. LangChain Models
- **LLMs** — Basic text completion with LLM wrappers
- **Chat Models** — Conversational models via OpenAI, Gemini, and HuggingFace (API & local)
- **Embedding Models** — Text embeddings for semantic search and document similarity

### 2. LangChain Prompts
- Static and dynamic prompt templates
- Building a chatbot with persistent chat history
- `ChatPromptTemplate` and `MessagesPlaceholder` for multi-turn conversations

### 3. Structured Output
- `with_structured_output()` using TypedDict, Pydantic models, and JSON schemas
- Output Parsers: `StrOutputParser`, `JsonOutputParser`, `PydanticOutputParser`

### 4. LangChain Chains
- `SimpleChain` → single-step pipelines
- `SequentialChain` → multi-step, output-to-input chaining
- `ParallelChain` → run multiple chains concurrently
- `ConditionalChain` → route to different chains based on logic

### 5. LangChain Runnables (LCEL)
- The modern way to compose LangChain pipelines using **LangChain Expression Language**
- `RunnableSequence`, `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda`, `RunnableBranch`

### 6. RAG (Retrieval-Augmented Generation)
- **Document Loaders** — Load from `.txt`, `.pdf`, directories, web URLs, and `.csv`
- **Text Splitters** — Split by length, text structure, document structure, or semantic similarity
- **Vector DB** — Store and query embeddings using ChromaDB
- **Retrievers** — Wikipedia, VectorStore similarity search, and MMR (Maximal Marginal Relevance)
- **YouTube Chatbot** — End-to-end RAG application using YouTube transcript data

### 7. Tools
- Creating custom tools in LangChain
- Binding tools to LLMs for autonomous tool calling
- Currency Conversion Tool as a practical example

---

## 🤝 Contributing

This is a learning repository. Feel free to fork it, open issues, or submit PRs to improve examples or add new topics.

---

## License

This project is open source and available under the [MIT License](LICENSE).