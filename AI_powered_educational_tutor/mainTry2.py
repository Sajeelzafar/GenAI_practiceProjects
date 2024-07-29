import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, initialize_agent, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key="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",
    model="llama3-70b-8192"
)

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
embeddings = OllamaEmbeddings(model="all-minilm")

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/react-chat")

retriever = db.as_retriever(
    search_type="similarity",
    # search_kwargs={"k": 3},
)

chat_history=[]

def get_subject_description(input_text):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a document reader specialist. Your task identify the main subject, and list any "
                "relevant headings, subheadings from the retrieved context ONLY in an educational context."
                "If the input is related to mathematics, focus on mathematical "
                "concepts, theorems, or branches that are relevant. List down your findings. If the subject is not found in document do not provide headings or sub headings. Ask the next bot to display the content as it is not the summary"),
            ("human", "{input}\n\nContext: {context}"),
        ]
    )
    context = retriever.get_relevant_documents(input_text)

    # Create the combined chain using LangChain Expression Language (LCEL)
    chain = prompt_template | llm | StrOutputParser()
    result = chain.invoke({"input": input_text, "context": context, "chat history": chat_history})
    return result

# List of tools available to the agent
tools = [
    Tool(
        name="Subject Tool",
        func=get_subject_description,
        description="Useful for when you need to know about a specific subject",
    ),
]

# Construct the ReAct agent
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": query, "chat_history": chat_history})
    print("Bot:", response["output"]) 
    
    # Update the chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=response["output"]))