import os
import streamlit as st
import PyPDF2  # PyPDF2 for PDF handling
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import subprocess

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from a PDF
def extract_text_from_pdf(uploaded_file):
    # Using the file-like object directly from Streamlit's uploader
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Function to ask a question using Ollama with the Mistral model
def ask_ollama(prompt, model="mistral"):
    try:
        result = subprocess.run(
            ["ollama", "run", model],  # Running the Mistral model
            input=prompt,
            text=True,
            capture_output=True,
            encoding='utf-8',
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        st.error(f"An error occurred: {e}")
        return None

# Set up the Streamlit UI
st.title("Study Assistant: Ask Questions from Your Book")

# File uploader to upload PDF files
uploaded_files = st.file_uploader("Upload your textbook (PDF)", type="pdf", accept_multiple_files=False)

# Text input for the student's question
query_text = st.text_input("What do you want to know from the book?")

# Text area for any additional input or custom instruction for the LLM
prompt_text = st.text_area("Add any specific instructions (optional):")

# Button to perform the search and generate a response
if st.button("Ask Question"):
    if uploaded_files and query_text:
        # Initialize FAISS Index
        dimension = model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        document_texts = []  # To store document texts
        document_metadata = []  # To store metadata, e.g., filenames

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_files)
        embedding = model.encode(text)
        index.add(np.array([embedding]))  # Add embedding to FAISS index
        document_texts.append(text)  # Save the extracted text
        document_metadata.append(uploaded_files.name)  # Save metadata, e.g., filename

        # Perform the similarity search
        query_embedding = model.encode(query_text)
        distances, indices = index.search(np.array([query_embedding]), k=3)  # Retrieve top 3 similar sections

        # Display the results
        st.write("**Relevant sections from the book:**")
        for i, idx in enumerate(indices[0]):
            #st.write(f"### Section {i+1}: {document_metadata[0]}")
           #st.write(f"**Content:**\n{document_texts[0][idx:idx + 500]}")  # Print a relevant portion of text
            #st.write(f"**Distance:** {distances[0][i]}")
            
            # Combine the prompt with the retrieved context
            context = document_texts[0][idx:idx + 500]  # Use the relevant section
            combined_input = f"Context: {context}\n\nPrompt: {prompt_text if prompt_text else query_text}"
            
        # Generate a response using Ollama's Mistral model
        response = ask_ollama(combined_input, model="mistral")  # Running Mistral model here
        st.write(f"**Answer from the book:**\n{response}")
        st.write("\n---\n")

    else:
        st.warning("Please upload a PDF of your book and ask a question.")
