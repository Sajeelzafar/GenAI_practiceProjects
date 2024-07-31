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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm = ChatGroq(
    api_key="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",
    model="llama3-70b-8192"
)

# Define Custom Input
custom_input_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assistant is a large language model trained by OpenAI.Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand."
    "Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics."
    "Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist."
    "When the assistant receives the response from a particular tool it uses. Assistant should also show the response of "
    "the tool along with its comment in the Final Answer."
    "TOOLS:"
    "------"
    "Assistant has access to the following tools:"
    "{tools}"
    "To use a tool, please use the following format:"
    "```"
    "Thought: Do I need to use a tool? Yes"
    "Action: the action to take, should be one of [{tool_names}]"
    "Action Input: the input to the action"
    "Observation: the result of the action"
    "```"
    "When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:"
    "```"
    "Thought: Do I need to use a tool? No"
    "Final Answer: [your response here]"
    "```"

    "When you receive a response from the tool to the Human, you MUST use the format:"
    "```"
    "Final Answer: [tool response here]"
    "```"
    
    "Begin!"

    "Previous conversation history:"
    "{chat_history}"

    "New input: {input}"
    "{agent_scratchpad}"
    )
])

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

def get_course_content(input_text):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "I want to learn about {input}. Please provide a detailed list of course contents and relevant links to study along with each content. If the subheading is not listed in previous chat history by the system, please inform me that this course is not available."),
            ("human", "Context: {context}"),
        ]
    )
    context = retriever.get_relevant_documents(input_text)
    # Create the combined chain using LangChain Expression Language (LCEL)
    chain = prompt_template | llm | StrOutputParser()
    result = chain.invoke({"input": input_text, "context": context, "chat history": chat_history})
    chat_history.append(SystemMessage(content=result))
    return result

def get_subject_description(input_text):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a document reader specialist. Your task identify the main subject, and list any "
                "relevant headings, subheadings from the retrieved context ONLY in an educational context."
                "If the input is related to mathematics, focus on mathematical "
                "concepts, theorems, or branches that are relevant. List down your findings. If the subject is not found in document do not provide headings or sub headings."),
            ("human", "{input}\n\nContext: {context}"),
        ]
    )
    context = retriever.get_relevant_documents(input_text)

    # Create the combined chain using LangChain Expression Language (LCEL)
    chain = prompt_template | llm | StrOutputParser()
    result = chain.invoke({"input": input_text, "context": context, "chat history": chat_history})
    chat_history.append(SystemMessage(content=result))
    return result

# List of tools available to the agent
tools = [
    Tool(
        name="Subject Tool",
        func=get_subject_description,
        description="Useful for when you need to know about a specific subject",
    ),
    Tool(
        name="Course Content Tool",
        func=get_course_content,
        description="Useful for when user has selected a heading or subheading from the previous responses only and asked about course content."
    )
]

# Construct the ReAct agent
agent = create_react_agent(llm, tools, custom_input_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": query, "chat_history": chat_history})
    # print("Bot:", response["output"]) 
    # print("---------Chat history------------")
    # Update the chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=response["output"]))
    # print(chat_history)(*args)