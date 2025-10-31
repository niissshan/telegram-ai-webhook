# main.py
import os
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx

# Telegram bot token (from Render environment)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Set TELEGRAM_TOKEN in environment")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI(title="Telegram AI Webhook (Free Model)")

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None


# 🧠 Free Hugging Face model endpoint (no API key needed)
HF_MODEL_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"

async def call_ai_model(user_text: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(HF_MODEL_URL, json={"inputs": user_text})
            data = response.json()
            # Handle possible API formats
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "Sorry, I couldn’t generate a response.")
            elif isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]
            else:
                return "Sorry, I couldn’t understand the response."
    except Exception as e:
        return f"Error contacting AI model: {str(e)}"


async def send_message(chat_id: int, text: str):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


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
        await send_message(chat_id, "Sorry, I only handle text messages for now.")
        return {"ok": True}

    if len(text) > 2000:
        await send_message(chat_id, "Please send shorter messages (under 2000 chars).")
        return {"ok": True}

    reply = await call_ai_model(text)
    await send_message(chat_id, reply)

    return {"ok": True}
