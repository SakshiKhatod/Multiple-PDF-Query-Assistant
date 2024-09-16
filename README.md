# Multiple-PDF-Query-Assistant
Assistant to query your multiple pdf's

This project provides an interactive tool that allows users to upload multiple PDF documents and ask questions about their content using advanced vector search techniques. The application processes the PDFs, converts their content into vector embeddings, and retrieves relevant information based on user queries.

**Features**
Upload multiple PDFs at once.
Ask questions based on the contents of the uploaded PDFs.
Query content efficiently with Pinecone vector search.
Uses SentenceTransformer for text embedding and OpenAI models for interaction.
Responsive UI built with Streamlit.
Requirements
**Python 3.8 or higher**

The following Python libraries:

**streamlit
dotenv
PyPDF2
langchain
sentence-transformers
faiss
numpy
htmlTemplates
openai
Hugging face**

**Setup**
**1. Clone the Repository**

Copy code
git clone https://github.com/SakshiKhatod/Multiple-PDF-Query-Assistant.git
cd Multiple-PDF-Query-Assistant

**2. Install Dependencies**
Use the following command to install all required dependencies from the requirements.txt file:

Copy code
pip install -r requirements.txt

**3. Set Up HUGGING FACE**
Create an account on openai and Hugging face.
Retrieve your API key and region from their dashboard.
Create a .env file in the root of the project and add your respective API key:
bash

HUGGINGFACEHUB_API_TOKEN=your-huggingface-api-key

**4. Set Up OpenAI API (Optional for Chat-Based Responses)**

Create an account on OpenAI.
Get your API key from the OpenAI dashboard.
Add your OpenAI API key to the .env file:

Copy code
OPENAI_API_KEY=your-openai-api-key

**5. Run the Application**
You can run the application locally using Streamlit:
Copy code
streamlit run app.py

This command will open a local web server, and you can interact with the tool from your browser.

**6. Uploading PDFs and Querying**
Upload multiple PDFs using the sidebar in the application.
Ask questions in the input field, and the system will retrieve relevant chunks of text from the uploaded PDFs.

**7.Project Structure**

app.py: Main application file with the Streamlit UI and the PDF querying logic where you can upload only single pdf file (upto 200 MB file)
app2.py: Main application file with the Streamlit UI and the PDF querying logic where you can upload multiple pdf file (upto 200 MB each file)
htmlTemplates.py: Contains HTML templates for customizing the UI.
requirements.txt: Lists all the required dependencies.
.env: Store your API keys here (this file should not be committed to version control).

**8.Usage
Example**

Upload PDFs: Use the sidebar to upload one or more PDF files.
Ask a Question: Enter a question in the text box, and the assistant will search the content of the uploaded PDFs to find the most relevant information.
Receive Responses: The assistant will respond based on the content of the uploaded PDFs, showing relevant text segments.

**9.Sample Queries**
"What is the main idea of the document?"
"Explain vector databases."
"How can AI be applied in healthcare?"

**10.Troubleshooting**
Missing OPENAIT API Key: Ensure the .env file contains your OPENAI API key.
Slow Queries: The application might take a few seconds to process large PDFs.
Incorrect Answers: Try refining your query for better results. The quality of responses depends on the quality of the document and embeddings.

**11. Contributing**
Contributions to improve the project are welcome! To contribute:
-Fork the repository.
-Create a new branch with your feature or bug fix.
-Submit a pull request, and ensure the tests pass.




