import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Legal Document QA System", layout="wide")
st.title("📑 Legal Document QA System (Multi-PDF + Summarization)")

# =======================
# 1. File Upload (Multiple PDFs)
# =======================
uploaded_files = st.file_uploader("Upload one or more legal documents (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            file_path = tmp_file.name

        # Load and split each PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs.extend(text_splitter.split_documents(documents))

    st.success(f"Loaded and split {len(docs)} chunks from {len(uploaded_files)} documents.")

    # =======================
    # 2. Preview Chunks
    # =======================
    st.write("#### Preview first 3 chunks:")
    for i, doc in enumerate(docs[:3]):
        st.write(f"**Chunk {i+1}:** {doc.page_content[:300]}...")

    # =======================
    # 3. Create Vector Store
    # =======================
    @st.cache_resource
    def get_vectorstore(docs):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return Chroma.from_documents(docs, embeddings)

    vectordb = get_vectorstore(docs)
    st.success("Vector store created!")

    # =======================
    # 4. Initialize Gemini LLM
    # =======================
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # =======================
    # 5. Summarization Prompt
    # =======================
    summary_prompt = PromptTemplate(
        template="""
You are a helpful legal assistant.
Summarize the following legal document in **plain, easy-to-understand language** for a non-lawyer.

Document Content:
{document}

Summary:""",
        input_variables=["document"]
    )

    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

    # Generate summaries
    with st.expander("📄 Document Summaries"):
        for idx, uploaded_file in enumerate(uploaded_files):
            chunk_text = "\n".join([doc.page_content for doc in docs[idx*len(docs)//len(uploaded_files):(idx+1)*len(docs)//len(uploaded_files)]])
            try:
                summary = summary_chain.run({"document": chunk_text})
                st.write(f"**{uploaded_file.name}:**")
                st.write(summary)
            except Exception as e:
                st.error(f"Error summarizing {uploaded_file.name}: {e}")

    # =======================
    # 6. RetrievalQA Prompt
    # =======================
    qa_prompt = PromptTemplate(
        template="""
You are a helpful legal assistant.
Answer the question using the context provided.
Explain in **simple terms for a non-lawyer**.

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": qa_prompt}
    )

    # =======================
    # 7. Ask Questions
    # =======================
    query = st.text_input("Ask a question about the uploaded documents:")

    if query:
        try:
            result = qa.invoke(query)
            st.write("### Answer:", result["result"])
        except Exception as e:
            st.error(f"Error: {e}")
