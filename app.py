import streamlit as st
from pdfreader import extract_text_from_pdf  # Make sure this works or use PyMuPDF / PyPDF2
from qachain import create_vector_store, get_most_similar_chunk, ask_gemini_continuous

import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyCAF5oWDxrm0uQsFCbC7FlBASaaUMdxb3s"  # Your Gemini API key

st.set_page_config(page_title="DocuQuery", layout="centered")
st.title(" PDF Q&A with Google Gemini")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)

    st.success("PDF text extracted!")

    with st.spinner("Creating vector index..."):
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        index, embeddings, chunk_list = create_vector_store(chunks)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("Searching and querying Gemini..."):
            matched_chunks = get_most_similar_chunk(query, index, embeddings, chunk_list)
            context = "\n\n".join(matched_chunks)
            answer = ask_gemini_continuous(query, context)
            st.session_state.chat_history.append({"question": query, "answer": answer})

    if st.session_state.chat_history:
        st.subheader(" Chat History")
        for qa in st.session_state.chat_history:
            st.markdown(f"**You:** {qa['question']}")
            st.markdown(f"**Gemini:** {qa['answer']}")
