import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain.schema import AIMessage, HumanMessage, SystemMessage
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

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Define the QA system prompt as a ChatPromptTemplate
qa_system_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a document reader specialist. Skim over the retrieved context and identify the subject in the file. "
                "List down the headings and subheadings of the context that matches the given input."
            )
        ),
    ]
)

# Define a very simple tool function that returns the current time
def get_subject_description():
    """Returns the subject categories and subcategories"""
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=qa_system_prompt_template,
    )

    result = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, handle_parsing_errors=True,
    )

    # Convert input to the required format
    return result

# List of tools available to the agent
tools = [
    Tool(
        name="Subject Tool",
        func=get_subject_description,
        description="Useful for when you need to know about a specific subject",
    ),
]

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history = []

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Define a prompt to determine the nature of the query
    determine_course_query_prompt = (
        "You are a highly knowledgeable and experienced AI. Determine if the user's input is a request for course content. "
        "If it is, respond with 'Course Query Detected'. Otherwise, continue the conversation as usual."
    )

    # Get AI response to determine if it's a course query
    system_message = SystemMessage(content=determine_course_query_prompt)
    result = llm.invoke([system_message, HumanMessage(content=query)])
    response = result.content

    if "Course Query Detected" in response:
        # Use the subject description tool
        course_response = get_subject_description(query)
        chat_history.append(AIMessage(content=course_response))
        print(f"AI: {course_response}")
    else:
        # Continue conversation as usual
        chat_history.append(AIMessage(content=response))
        print(f"AI: {response}")
