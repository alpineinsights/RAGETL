import streamlit as st
import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from voyageai import Client as VoyageClient
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
import pinecone
import tempfile

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
voyage_api_key = os.getenv("VOYAGEAI_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
llamaparse_key = os.getenv("LLAMA_CLOUD_API_KEY")

# Initialize clients
voyage_client = VoyageClient(api_key=voyage_api_key)
anthropic_client = Anthropic(api_key=anthropic_api_key)
llama_parser = LlamaParse(api_key=llamaparse_key, result_type="text")

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")

# Initialize session state variables
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 512
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 256
if 'parsing_instructions' not in st.session_state:
    st.session_state.parsing_instructions = ""

def parse_document(file, parsing_instructions):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.getvalue())
        temp_file_path = temp_file.name
    
    parsed_text = llama_parser.parse_file(temp_file_path, parsing_instructions=parsing_instructions)
    os.unlink(temp_file_path)
    return parsed_text

def chunk_text(text, chunk_size, chunk_overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap
    return chunks

def create_sparse_vectors(chunks):
    vectorizer = TfidfVectorizer()
    sparse_vectors = vectorizer.fit_transform(chunks)
    return sparse_vectors, vectorizer

def create_dense_vectors(chunks):
    dense_vectors = voyage_client.embed(chunks, model="voyage-finance-2").embeddings
    return dense_vectors

def store_in_pinecone(chunks, sparse_vectors, dense_vectors, index_name):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(dense_vectors[0]), metric="cosine")
    
    index = pinecone.Index(index_name)
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = list(zip(chunks[i:i+batch_size], sparse_vectors[i:i+batch_size], dense_vectors[i:i+batch_size]))
        to_upsert = [
            (str(j), dense_vector, {"text": chunk, "sparse_vector": sparse_vector.toarray()[0].tolist()})
            for j, (chunk, sparse_vector, dense_vector) in enumerate(batch, start=i)
        ]
        index.upsert(vectors=to_upsert)

def process_document(file, parsing_instructions, chunk_size, chunk_overlap):
    text = parse_document(file, parsing_instructions)
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    sparse_vectors, vectorizer = create_sparse_vectors(chunks)
    dense_vectors = create_dense_vectors(chunks)
    store_in_pinecone(chunks, sparse_vectors, dense_vectors, pinecone_index_name)
    return vectorizer

def hybrid_search(query, index_name, top_k=5, alpha=0.5):
    index = pinecone.Index(index_name)
    
    # Dense search
    dense_query = voyage_client.embed([query], model="voyage-finance-2").embeddings[0]
    dense_results = index.query(dense_query, top_k=top_k, include_metadata=True)
    
    # Sparse search
    sparse_query = vectorizer.transform([query]).toarray()[0]
    sparse_results = index.query(
        vector=[0] * len(dense_query),  # Dummy dense vector
        sparse_vector=dict(enumerate(sparse_query)),
        top_k=top_k,
        include_metadata=True
    )
    
    # Combine results
    combined_results = {}
    for result in dense_results['matches'] + sparse_results['matches']:
        id = result['id']
        if id not in combined_results:
            combined_results[id] = {
                'id': id,
                'text': result['metadata']['text'],
                'score': 0
            }
        combined_results[id]['score'] += result['score'] * (alpha if result in dense_results['matches'] else (1-alpha))
    
    # Sort and return top results
    return sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)[:top_k]

def main():
    st.title("Document Processing and Hybrid Search")

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # ETL Parameters
    st.header("ETL Parameters")
    chunk_size = st.number_input("Chunk size", min_value=100, value=st.session_state.chunk_size)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, value=st.session_state.chunk_overlap)
    parsing_instructions = st.text_area("Parsing Instructions", value=st.session_state.parsing_instructions)

    # Process document
    if uploaded_file is not None and st.button("Process Document"):
        with st.spinner("Processing document..."):
            global vectorizer
            vectorizer = process_document(uploaded_file, parsing_instructions, chunk_size, chunk_overlap)
        st.success("Document processed and stored in Pinecone!")

    # Search
    st.header("Hybrid Search")
    query = st.text_input("Enter your search query")
    if query and st.button("Search"):
        with st.spinner("Searching..."):
            results = hybrid_search(query, pinecone_index_name)
        
        st.subheader("Search Results")
        for result in results:
            st.write(f"Score: {result['score']:.4f}")
            st.write(result['text'])
            st.write("---")

if __name__ == "__main__":
    main()
