📑 Legal Document QA System (Multi-PDF + Summarization)

This Streamlit app allows users to upload multiple legal PDF documents and interact with them using AI-powered summarization and question answering. It leverages LangChain, Chroma vector store, HuggingFace embeddings, and Google’s Gemini (via LangChain) for natural language processing.

🔹 Key Features:
      Upload Multiple PDFs – Users can upload one or more legal documents in PDF format.
      Text Processing – Each PDF is split into smaller, manageable text chunks for better retrieval.
      Vector Database (Chroma) – All document chunks are embedded and stored in a vector database for semantic search.
      Summarization – Automatically generates easy-to-understand summaries of each document using Gemini and a custom prompt.
      Preview Chunks – Displays the first few text chunks to verify document processing.
      Question Answering – Users can ask natural language questions about the uploaded documents. The system retrieves relevant chunks and provides simplified, lawyer-friendly answers.
      Google Gemini LLM – Uses gemini-1.5-flash for fast, accurate, and context-aware responses.

🔹 Tech Stack:
    Streamlit → Web app interface
    LangChain → Document loading, text splitting, prompts, and QA chains
    ChromaDB → Vector storage for document embeddings
    HuggingFace Embeddings → all-MiniLM-L6-v2 model for semantic similarity
    Google Generative AI (Gemini) → Summarization & Q&A
    PyPDFLoader → Extract text from PDFs

🔹 Use Cases:
    Reviewing contracts and agreements
    Extracting key clauses without legal jargon
    Quickly answering compliance-related questions
    Summarizing lengthy legal documents for non-lawyers
