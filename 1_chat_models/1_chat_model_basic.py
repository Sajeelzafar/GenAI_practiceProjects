# from langchain_community.llms import Ollama

# model = Ollama(model="llama3")
import os

from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "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"
GROQ_LLM = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192"
        )

# Invoke the model with a message
result = GROQ_LLM.invoke("What is 81 divided by 9?")
print(result.content)