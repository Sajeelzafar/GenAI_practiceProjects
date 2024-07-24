import os

from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
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

# Define the user's question
query = "Who is Minerva?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    # search_type="similarity_score_threshold",
    # search_kwargs={"k": 3, "score_threshold": 0.01},
)
relevant_docs = retriever.invoke(query)

system = "You are a helpful assistant. Read the provided documents and answer the query in one or two lines"
human = "{query}, {relevant_docs}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
chain = prompt | chat

airesponse = chain.invoke({"query": query, "relevant_docs": relevant_docs})
print("\n--- AI Response ---")
print(airesponse)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
