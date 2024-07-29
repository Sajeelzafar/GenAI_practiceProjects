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

chat_history=[]

# Create a structured Chat Agent with Conversation Buffer Memory
# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# Define a prompt to determine the nature of the query
determine_course_query_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are an educational assistant. Your task is to help users with their queries about various subjects. Use the Subject Tool when you need specific information about a subject. Provide concise and informative responses."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessage(content="{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

def get_subject_description(input_text):
    """Returns the subject categories and subcategories"""
    try:
        print("Input text is:", input_text)
        # Retrieve relevant documents from the Chroma database
        docs = db.similarity_search(input_text, k=3)
        secondaryPrompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a document reader specialist. Your task is to analyze the following input text "
                "as if it were part of a larger document. Identify the main subject, and list any relevant "
                "headings, subheadings, or key concepts that would likely be associated with this topic "
                "in an educational context. If the input is related to mathematics, focus on mathematical "
                "concepts, theorems, or branches that are relevant. Provide a concise summary of your findings."),
            HumanMessage(content="{input}\n\nContext: {context}")
        ])
        messages = secondaryPrompt.format_messages(input=input_text)
        result = llm.invoke(messages)
        print("----Result----")
        print(result.content)
        return result.content
    except Exception as e:
        print(f"Error in get_subject_description: {e}")
        return "An error occurred while processing your request."

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


# AgentExecutor is responsible for managing the interaction between the user input, the agent, and the tools
# It also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,  # Handle any parsing errors gracefully
)

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

    # Add the user's message to the conversation memory
    # memory.chat_memory.add_message(HumanMessage(content=query))
    # Add the agent's response to the conversation memory
    # memory.chat_memory.add_message(AIMessage(content=response["output"]))
    

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
