import google.generativeai as genai
import faiss
import numpy as np
genai.configure(api_key="AIzaSyCAF5oWDxrm0uQsFCbC7FlBASaaUMdxb3s")

def embed_texts_with_gemini(text_list):
    embeddings = [
        genai.embed_content(
            model="models/embedding-001",
            content=txt,
            task_type="retrieval_document"
        )['embedding']
        for txt in text_list
    ]
    return embeddings

def create_vector_store(chunks):
    embeddings = embed_texts_with_gemini(chunks)
    embeddings_np = np.array(embeddings).astype("float32")

    # Create FAISS index with L2 distance
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)

    return index, embeddings, chunks

def get_most_similar_chunk(query, faiss_index, original_embeddings, chunks, top_k=1):
    """
    Returns the top_k most similar text chunks to the query.
    """
   
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )['embedding']

    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = faiss_index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

def ask_gemini_continuous(query, context):
    prompt = f"""You are a helpful assistant. Use the following context from a PDF to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""

    model = genai.GenerativeModel("models/gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()
