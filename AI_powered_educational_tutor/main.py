import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent, initialize_agent, create_react_agent
from langchain.memory import ConversationBufferMemory
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

# Load the correct JSON Chat Prompt from the hub
prompt = hub.pull("hwchase17/structured-chat-agent")

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a structured Chat Agent with Conversation Buffer Memory
# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# Define the QA system prompt as a ChatPromptTemplate
secondary_agent_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a document reader specialist. Skim over the retrieved context and identify the subject in the file. "
                "List down the headings and subheadings of the context that matches the given input."
            )
        ),
    ]
)

# Define a prompt to determine the nature of the query
determine_course_query_prompt = (
    "You are an educational content creator agent. Your task is to classify queries into two categories:" "subject-specific and general. Follow these steps:"

    "1. Identify if the query pertains to a specific subject (e.g., math, history, biology) or if it is "
    "a general inquiry."
    "2. If the query is subject-specific, classify it into the appropriate subject category."
    "3. Provide a brief explanation of your classification decision."
)

# Define a very simple tool function that returns the current time
def get_subject_description(input_text):
    """Returns the subject categories and subcategories"""
    print("input_text", input_text)
    # agent = create_react_agent(
    #     llm=llm,
    #     prompt=qa_system_prompt_template,
    #     verbose=True
    # )
    secondary_agent = create_react_agent(
        llm=llm,
        prompt=secondary_agent_prompt,
        verbose=True
    )
    secondary_agent_executor = AgentExecutor(agent=secondary_agent, tools=[], verbose=True)
    
    # Use the secondary agent to perform the search
    result = secondary_agent_executor.invoke({"input": input_text})
    
    print("----Result----")
    print(result)
    return result

    memory.chat_memory.add_message(HumanMessage(content=query))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": query})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))

    # subject_prompt = ChatPromptTemplate
    return

    # result = AgentExecutor.from_agent_and_tools(
    #     agent=agent, tools=tools, handle_parsing_errors=True,
    # )

    # Convert input to the required format
    # return result

# List of tools available to the agent
tools = [
    Tool(
        name="Subject Tool",
        func=get_subject_description,
        description="Useful for when you need to know about a specific subject",
    ),
]

# create_structured_chat_agent initializes a chat agent designed to interact using a structured prompt and tools
# It combines the language model (llm), tools, and prompt to create an interactive agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# system_message = SystemMessage(content="You are a helpful AI assistant.")
# chat_history = []

# AgentExecutor is responsible for managing the interaction between the user input, the agent, and the tools
# It also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  # Use the conversation memory to maintain context
    handle_parsing_errors=True,  # Handle any parsing errors gracefully
)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    # chat_history.append(HumanMessage(content=query))  # Add user message
    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=query))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": query})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
    

    # Get AI response to determine if it's a course query
    # system_message = SystemMessage(content=determine_course_query_prompt)
    # result = llm.invoke([system_message, HumanMessage(content=query)])
    # response = result.content

    # if "Course Query Detected" in response:
    #     # Use the subject description tool
    #     course_response = get_subject_description(query)
    #     chat_history.append(AIMessage(content=course_response))
    #     print(f"AI: {course_response}")
    # else:
    #     # Continue conversation as usual
    #     chat_history.append(AIMessage(content=response))
    #     print(f"AI: {response}")
