# main.py
import os
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Set TELEGRAM_TOKEN in environment")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI(title="Free Telegram AI Chatbot")

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None


async def call_free_ai(user_text: str) -> str:
    """
    Calls Hugging Face's public GPT-2 model (no API key needed)
    """
    url = "https://api-inference.huggingface.co/models/gpt2"
    payload = {"inputs": f"User: {user_text}\nAI:"}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            data = resp.json()

            # Handle if model is cold-starting
            if isinstance(data, dict) and "error" in data and "loading" in data["error"].lower():
                return "The AI model is waking up — please try again in a few seconds."

            # Extract text
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                reply = data[0]["generated_text"].split("AI:")[-1].strip()
                return reply or "I'm here! How can I help you today?"
            else:
                return "I'm here! How can I help you today?"

    except Exception as e:
        return f"Temporary issue contacting AI service: {str(e)}"


async def send_message(chat_id: int, text: str):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(url, json=payload)


@app.post("/webhook")
async def telegram_webhook(update: Message, request: Request):
    data = await request.json()
    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"ok": True, "note": "no message"}

    chat = message.get("chat", {})
    chat_id = chat.get("id")
    text = message.get("text") or message.get("caption") or ""
    if not text.strip():
        await send_message(chat_id, "Please send text only 🙂")
        return {"ok": True}

    reply = await call_free_ai(text)
    await send_message(chat_id, reply)
    return {"ok": True}
