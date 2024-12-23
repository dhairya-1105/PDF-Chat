import streamlit as st
import PyPDF2
import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
import subprocess

def query_llama(prompt: str):
    """ Function to query LLaMA 3.2 model using Ollama CLI. """
    try:
        # Run the Ollama model with the prompt directly in the command
        result = subprocess.run(
            ["ollama", "run", "llama3.2", prompt],
            capture_output=True,
            text=True
        )
        # Check if there is any error in the result
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        # Debug: print the raw output from the model
        print("Raw output from LLaMA:")
        print(result.stdout)

        # Try parsing the result
        try:
            response = result.stdout.strip()  # Remove any extra spaces or newlines
            if response:
                return response
            else:
                return "No valid response from the model."
        except Exception as e:
            return f"Error: Failed to process the response from LLaMA. {str(e)}"
    
    except Exception as e:
        return f"Error: {e}"

with st.sidebar:
    st.title('LLM Chat App')

def main():
    st.header("Chat with PDFs")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf is not None:
        st.write(pdf.name)

        pdf_reader = PyPDF2.PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            instructor_embeddings = HuggingFaceInstructEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2", 
                model_kwargs={"device": "cuda"}
            )
            VectorStore = FAISS.from_texts(chunks, instructor_embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query = st.text_input("Enter a query:")

        if query:
            # Retrieve relevant documents using similarity search
            docs = VectorStore.similarity_search(query=query, k=3)  # Retrieving top 1 relevant document
            if docs:
                context = " ".join([doc.page_content for doc in docs])  # Use .page_content to access text
                prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
                answer = query_llama(prompt)
                st.write("Answer:")
                st.write(answer)
            else:
                st.write("No relevant documents found.")

if __name__ == '__main__':
    main()
