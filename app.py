from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # Upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Generate embeddings
        embeddings = model.encode(chunks)

        # Create FAISS index (proper method)
        knowledge_base = FAISS.from_texts(
            texts=chunks,
            embedding=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )

        # Load a question-answering pipeline
        qa_model = pipeline(
            "question-answering", model="distilbert-base-uncased-distilled-squad"
        )

        # Show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question, k=5)
            context = " ".join([doc.page_content for doc in docs])
            # st.write(context)
            # # Truncate context to 200 tokens
            # context_tokens = context.split()[:200]
            # truncated_context = " ".join(context_tokens)

            # st.write(truncated_context)

            # Generate the answer based on the truncated context using the QA model
            response = qa_model(question=user_question, context=context)
            # st.write(response)

            # Display the answer
            st.write("**Answer:**", response["answer"])


if __name__ == "__main__":
    main()
