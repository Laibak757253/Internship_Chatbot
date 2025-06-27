import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# === Environment Variables ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

# === Other Constants ===
CHARLES_PROMPT = """You are Charles, an intelligent assistant with internet search capabilities through Tavily.

Use the internet search when:
- Answering questions about the current date, news, weather, etc.
- The user asks about something "today", "now", "latest", etc.

Avoid using the tool for timeless information like math, code, definitions, or opinions."""
