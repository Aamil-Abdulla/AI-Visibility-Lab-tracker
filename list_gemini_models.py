"""Prints the model IDs your own Gemini key can use. Pick one for GEMINI_MODEL in .env."""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions:
        print(m.name.removeprefix("models/"))
