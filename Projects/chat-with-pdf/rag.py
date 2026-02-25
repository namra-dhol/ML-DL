# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    formatted = "\n\n".join(doc.page_content for doc in docs)
    print(f"\n{'='*50}")
    print(f"RETRIEVED {len(docs)} CHUNKS FROM PDF:")
    print(f"{'='*50}")
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
    print(f"{'='*50}\n")
    return formatted

def load_rag_chain(chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # creating vector database
    vector_db = Chroma.from_texts(texts=chunks, embedding=embedding_model)

    # Use MMR (Maximal Marginal Relevance) for better diversity
    # Retrieve more chunks (10 instead of 5) for better context
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 20}
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a PDF document assistant. Your ONLY job is to answer questions based STRICTLY on the context provided below from the PDF document.
    
    CRITICAL RULES:
    1. You MUST answer ONLY using information from the Context below
    2. If the answer is NOT in the Context, you MUST say "I cannot find this information in the uploaded PDF"
    3. DO NOT use your general knowledge - ONLY use the Context provided
    4. Quote or reference specific parts from the Context when answering
    5. If the Context is empty or irrelevant, say "The PDF does not contain information about this topic"
    
    Context from PDF:
    {context}
    
    User Question:
    {question}
    
    Answer (based ONLY on the Context above):
    """)
    llm = OllamaLLM(model="gpt-oss:20b-cloud")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain
