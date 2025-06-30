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

# Langfuse
from langfuse import Langfuse 
from langfuse import observe 
from uuid import uuid4

# APIs
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# connect to mongodb
client = MongoClient(os.getenv("MONGODB_URI"))
db = client.chatbot_db
chat_collection = db.chats

# simple memory to remember name during session
temp_memory = {"name": None}

# define langgraph state type
class State(TypedDict):
    messages: Annotated[list, add_messages]

# define internet search tool
@tool
def tavily_search(query: str) -> str:
    """search the internet for real-time info"""
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = client.search(query=query, max_results=2)
    return str(results)

# setup tools and llm
tools = [tavily_search]
llm = init_chat_model("google_genai:gemini-2.0-flash")
llm_with_tools = llm.bind_tools(tools)

# base system prompt
BASE_PROMPT = """you are charles, a smart assistant with internet search using tavily.

use search for current info like:
- date, news, weather, or when user says "today", "now", etc.

don't use search for facts, math, code, or opinions."""

# langfuse-traced chatbot node
@observe(name="LLM Chatbot")
def chatbot(state: State, **kwargs):
    trace = kwargs.get("trace")
    if trace:
        trace.set_user_id("charles-user")

    messages = state["messages"]

    # extract user's name if they say "my name is"
    if isinstance(messages[-1], HumanMessage):
        content = messages[-1].content.lower()
        if "my name is" in content:
            try:
                name = content.split("my name is")[-1].strip().split()[0]
                temp_memory["name"] = name.capitalize()
            except:
                pass

    # add system message
    system_content = BASE_PROMPT
    if temp_memory["name"]:
        system_content += f"\n\nuser's name is {temp_memory['name']}."

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=system_content)] + messages
    else:
        messages[0] = SystemMessage(content=system_content)

    # call the llm
    output = llm_with_tools.invoke(messages)
    return {"messages": [output]}

# build the langgraph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

# stream response and log to mongodb
def stream_graph_updates(user_input: str):
    response = ""
    message_doc = {"question": user_input, "answer": None}

    for event in graph.stream({"messages": [HumanMessage(content=user_input)]}):
        for value in event.values():
            message = value["messages"][-1]
            if hasattr(message, 'content') and isinstance(message.content, str) and not isinstance(message, ToolMessage):
                response = message.content
                yield response

    # save to mongodb
    message_doc["answer"] = response
    chat_collection.insert_one(message_doc)

    # keep only the 10 latest messages
    if chat_collection.count_documents({}) > 10:
        oldest = chat_collection.find().sort("_id", 1).limit(1)[0]
        chat_collection.delete_one({"_id": oldest["_id"]})

# setup fastapi
app = FastAPI()

# allow requests from any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# input model for api
class ChatInput(BaseModel):
    message: str

# define chat endpoint
@app.post("/chat")
def chat_endpoint(chat: ChatInput):
    full_response = ""
    for chunk in stream_graph_updates(chat.message):
        full_response = chunk
    return {"response": full_response}

# gradio interface
if __name__ == "__main__":
    import gradio as gr

    def chat_with_bot(message, history):
        try:
            final_response = ""
            for chunk in stream_graph_updates(message):
                final_response = chunk
                yield final_response
            if not final_response:
                yield "sorry, i didn't get a response."
        except Exception as e:
            yield f"error: {str(e)}"

    demo = gr.ChatInterface(
        fn=chat_with_bot,
        title="my assistant charles",
        theme=gr.themes.Base(primary_hue="pink"),
        type="messages",
        save_history=True
    )

    demo.launch(server_name="127.0.0.1", server_port=7861, share=True, inbrowser=True)
