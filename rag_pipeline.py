from groq import Groq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import os

# Initialize Groq client (use environment variable for API key)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def setup_vector_store(text):
    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Create vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def query_rag(vector_store, query):
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    # System prompt for medical context
    system_prompt = "You are a medical assistant. Use the provided patient report context to answer queries accurately."
    full_prompt = f"Context: {context}\n\nQuery: {query}"
    
    # Prepare messages for Groq API
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_prompt}
    ]
    
    # Call Groq API
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=messages,
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    # Collect response from streaming chunks
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    
    return response