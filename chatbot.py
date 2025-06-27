import time
from typing import Annotated
from typing_extensions import TypedDict
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import tool

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# MongoDB Setup
client = MongoClient(os.getenv("MONGODB_URI"))
db = client.chatbot_db
chat_collection = db.chats

# Session memory for remembering user's name
temp_memory = {"name": None}

# Define the State
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Custom Tavily Tool
@tool
def tavily_search(query: str) -> str:
    """Use Tavily to search the internet for recent or real-time information."""
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=query, max_results=2)
    return str(results)

# Setup tools and LLM
tools = [tavily_search]
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

# Base system prompt
BASE_PROMPT = """You are Charles, an intelligent assistant with internet search capabilities through Tavily.

Use the internet search when:
- Answering questions about the current date, news, weather, etc.
- The user asks about something \"today\", \"now\", \"latest\", etc.

Avoid using the tool for timeless information like math, code, definitions, or opinions."""

# Chat node
def chatbot(state: State):
    messages = state["messages"]

    if isinstance(messages[-1], HumanMessage):
        content = messages[-1].content.lower()
        if "my name is" in content:
            try:
                name = content.split("my name is")[-1].strip().split()[0]
                temp_memory["name"] = name.capitalize()
            except:
                pass

    system_content = BASE_PROMPT
    if temp_memory["name"]:
        system_content += f"\n\nUser's name is {temp_memory['name']}."

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_content)] + messages
    else:
        messages[0] = SystemMessage(content=system_content)

    return {"messages": [llm_with_tools.invoke(messages)]}

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

# Stream response
def stream_graph_updates(user_input: str):
    response = ""
    message_doc = {"question": user_input, "answer": None}
    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        for value in event.values():
            message = value["messages"][-1]
            if hasattr(message, 'content') and isinstance(message.content, str) and not isinstance(message, ToolMessage):
                response = message.content
                yield response
    message_doc["answer"] = response
    chat_collection.insert_one(message_doc)
    if chat_collection.count_documents({}) > 10:
        oldest = chat_collection.find().sort("_id", 1).limit(1)[0]
        chat_collection.delete_one({"_id": oldest["_id"]})

# FastAPI Setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInput(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(chat: ChatInput):
    full_response = ""
    for chunk in stream_graph_updates(chat.message):
        full_response = chunk
    return {"response": full_response}

# Gradio Interface with Streaming and Threading
if __name__ == "__main__":
    import gradio as gr

    def chat_with_bot(message, history):
        try:
            final_response = ""
            for chunk in stream_graph_updates(message):
                final_response = chunk
                yield final_response
            if not final_response:
                yield "Sorry, I didn't get a response."
        except Exception as e:
            yield f"Error: {str(e)}"

    demo = gr.ChatInterface(
        fn=chat_with_bot,
        title="My Assistant Charles",
        theme=gr.themes.Base(primary_hue="pink"),
        type="messages",
        save_history=True
    )

    demo.launch(server_name="127.0.0.1", server_port=7861, share=True, inbrowser=True)

