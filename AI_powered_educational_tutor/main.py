import os

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
import os
from langchain_core.prompts import ChatPromptTemplate

from langchain_groq import ChatGroq

chat = ChatGroq(
            api_key="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",
            model="llama3-70b-8192"
        )

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
embeddings = OllamaEmbeddings(model="all-minilm")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)