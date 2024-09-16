import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.embeddings import HuggingFaceEmbeddings
from htmlTemplates import css, user_template, bot_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(text_chunks)
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    return vectorstore


def handle_userinput(user_question, vectorstore, qa_model):
    # Retrieve relevant documents from the vectorstore
    docs = vectorstore.similarity_search(user_question, k=5)
    context = " ".join([doc.page_content for doc in docs])

    # Use the QA model to get the answer
    response = qa_model(question=user_question, context=context)
    st.session_state.answer = response["answer"]

    # st.write(st.session_state.answer)
    # st.write(
    #     user_template.replace("{{MSG}}", user_question),
    #     unsafe_allow_html=True,
    # )
    st.write(
        bot_template.replace("{{MSG}}", st.session_state.answer), unsafe_allow_html=True
    )

    # Display the answer
    # st.write("**Answer:**", response["answer"])


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "qa_model" not in st.session_state:
        st.session_state.qa_model = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with multiple PDFs :books:")
    # Sidebar for file upload and processing
    with st.sidebar:
        st.subheader("Upload your PDFs")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here", accept_multiple_files=True, type="pdf"
        )

        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Load the question-answering pipeline
                    qa_model = pipeline(
                        "question-answering",
                        model="distilbert-base-uncased-distilled-squad",
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_model = qa_model
                    st.success("PDFs processed successfully!")

    # Main area for user input and displaying results
    user_question = st.text_input("Ask a question about your documents:")

    if (
        user_question
        and "vectorstore" in st.session_state
        and "qa_model" in st.session_state
    ):
        handle_userinput(
            user_question, st.session_state.vectorstore, st.session_state.qa_model
        )


if __name__ == "__main__":
    main()
