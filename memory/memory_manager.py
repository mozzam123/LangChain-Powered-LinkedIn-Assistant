from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("Groq API Key not found in environment !")


# Create the Groq LLM instance
def get_groq_llm(temperature = 0.0, model_name = "llama3-8b-8192"):
    return ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature
    )

# Unified Memory Loader
def get_memory(memory_type: str = "combined", k: int = 5):
    """
    Returns a memory instance based on the requested type.

    memory_type: "buffer", "summary", or "combined"
    k: number of recent messages to store in buffer
    """
    if memory_type == "buffer":
        return ConversationBufferWindowMemory(
            
            k=k,
            return_messages=True,
            memory_key="buffer_history"
        )

    elif memory_type == "summary":
        llm = get_groq_llm()
        return ConversationSummaryMemory(
            llm=llm,
            return_messages=True,
            memory_key="summary_history"
        )

    elif memory_type == "combined":
        llm = get_groq_llm()
        return CombinedMemory(
            memories=[
                ConversationBufferWindowMemory(k=k, return_messages=True, memory_key="buffer_history"),
                ConversationSummaryMemory(llm=llm, return_messages=True, memory_key="chat_history")
            ]
        )

    else:
        raise ValueError(f"Unknown memory type: {memory_type}")