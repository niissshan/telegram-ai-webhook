import os
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_API_KEY = os.getenv("HF_API_KEY")  # Hugging Face API key (optional)
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

if not TELEGRAM_TOKEN:
    raise RuntimeError("Set TELEGRAM_TOKEN in environment")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

app = FastAPI(title="Telegram AI Webhook")

# 🧠 Simple in-memory chat history (chat_id → list of messages)
chat_history: Dict[int, list[str]] = {}

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None


async def call_huggingface(chat_id: int, user_text: str) -> str:
    """Call Hugging Face model with short memory"""
    # Store user message
    history = chat_history.get(chat_id, [])
    history.append(f"User: {user_text}")

    # Keep only last 5 turns
    history = history[-10:]

    prompt = "\n".join(history) + "\nAI:"

    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150, "temperature": 0.7}
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(HF_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            # Extract text response
            text = ""
            if isinstance(data, list) and len(data) > 0:
                text = data[0].get("generated_text", "")
            if not text:
                text = "Sorry, I couldn’t generate a proper response."

            # Extract AI response (only new part)
            reply = text.split("AI:")[-1].strip()
            if not reply:
                reply = text.strip()

            # Save AI reply in memory
            history.append(f"AI: {reply}")
            chat_history[chat_id] = history

            return reply
        except Exception as e:
            print(f"❌ HuggingFace error: {e}")
            return "Sorry — I couldn’t reach the AI service right now."


async def send_message(chat_id: int, text: str):
    """Send a message to Telegram"""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


@app.post("/webhook")
async def telegram_webhook(update: Message, request: Request):
    """Handle Telegram webhook updates"""
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

    if text.lower() in ["/reset", "reset", "clear"]:
        chat_history.pop(chat_id, None)
        await send_message(chat_id, "🧹 Chat memory cleared.")
        return {"ok": True}

    reply = await call_huggingface(chat_id, text)
    await send_message(chat_id, reply)
    return {"ok": True}
