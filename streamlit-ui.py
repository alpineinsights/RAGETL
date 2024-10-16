import streamlit as st
import os
import tempfile
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from voyageai import Client as VoyageClient
import pinecone

# Set page config
st.set_page_config(page_title="Contextual RAG Pipeline", layout="wide")

# Title
st.title("Contextual RAG Pipeline")

# Sidebar for API keys
st.sidebar.header("API Keys")
llama_cloud_api_key = st.sidebar.text_input("LlamaIndex API Key", type="password")
anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
voyage_api_key = st.sidebar.text_input("Voyage AI API Key", type="password")

# Main content
st.header("Upload PDF Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Pipeline parameters
st.header("Pipeline Parameters")
chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2048, value=1024)
chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200)
embedding_model = st.selectbox("Embedding Model", ["voyage-finance-2", "voyage-2", "voyage-large-2"])
vector_index_name = st.text_input("Vector Index Name", "contextual-rag-index")

# Pipeline functions
def parse_pdf(file):
    parser = LlamaParse(result_type="text", api_key=llama_cloud_api_key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name

    documents = SimpleDirectoryReader(input_files=[temp_file_path], file_extractor={".pdf": parser}).load_data()
    os.unlink(temp_file_path)
    return documents[0].text

def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def generate_context(chunk, anthropic_client):
    prompt = f"""
    <document>
    {chunk}
    </document>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """
    message = anthropic_client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        temperature=0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content

def create_sparse_vectors(chunks):
    vectorizer = TfidfVectorizer()
    sparse_vectors = vectorizer.fit_transform(chunks)
    return sparse_vectors, vectorizer

def create_dense_vectors(chunks, voyage_client, model):
    dense_vectors = voyage_client.embed(chunks, model=model).embeddings
    return dense_vectors

def store_in_pinecone(chunks, contexts, sparse_vectors, dense_vectors, index_name):
    pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(dense_vectors[0]), metric="cosine")
    
    index = pinecone.Index(index_name)
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = list(zip(chunks[i:i+batch_size], contexts[i:i+batch_size], sparse_vectors[i:i+batch_size], dense_vectors[i:i+batch_size]))
        to_upsert = [
            (str(j), vector.tolist(), {"text": chunk, "context": context, "sparse_vector": sparse.toarray().tolist()[0]})
            for j, (chunk, context, sparse, vector) in enumerate(batch, start=i)
        ]
        index.upsert(vectors=to_upsert)

def process_document(file, chunk_size, chunk_overlap, embedding_model, vector_index_name):
    # Parse PDF
    text = parse_pdf(file)
    
    # Chunk text
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    # Generate context for each chunk
    anthropic_client = Anthropic(api_key=anthropic_api_key)
    contexts = [generate_context(chunk, anthropic_client) for chunk in chunks]
    
    # Create sparse vectors
    sparse_vectors, _ = create_sparse_vectors(chunks)
    
    # Create dense vectors
    voyage_client = VoyageClient(api_key=voyage_api_key)
    dense_vectors = create_dense_vectors(chunks, voyage_client, embedding_model)
    
    # Store in Pinecone
    store_in_pinecone(chunks, contexts, sparse_vectors, dense_vectors, vector_index_name)

    return len(chunks)

# Process button
if st.button("Process Document"):
    if uploaded_file is not None:
        if not all([llama_cloud_api_key, anthropic_api_key, pinecone_api_key, voyage_api_key]):
            st.error("Please provide all required API keys.")
        else:
            st.info("Processing document... This may take a while.")
            try:
                num_chunks = process_document(uploaded_file, chunk_size, chunk_overlap, embedding_model, vector_index_name)
                st.success(f"Document processed successfully! {num_chunks} chunks created and stored in Pinecone.")
            except Exception as e:
                st.error(f"An error occurred while processing the document: {str(e)}")
    else:
        st.error("Please upload a PDF document first.")
