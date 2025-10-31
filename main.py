# main.py
import os
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("Set TELEGRAM_TOKEN in environment")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI(title="Telegram AI Webhook (Free Model)")

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None

# Two fallback free models
PRIMARY_MODEL = "facebook/blenderbot-400M-distill"
FALLBACK_MODEL = "microsoft/DialoGPT-medium"

async def call_huggingface_model(user_text: str, model_name: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json={"inputs": user_text})
            data = resp.json()

            # Handle "loading" or queued models
            if isinstance(data, dict) and "error" in data and "loading" in data["error"].lower():
                return "The AI model is waking up. Please try again in a few seconds."

            # Extract generated text properly
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                return data[0]["generated_text"]
            elif isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"]

            return "Sorry, I couldn’t generate a proper response."
    except Exception as e:
        return f"Error contacting Hugging Face model: {str(e)}"

async def call_ai(user_text: str) -> str:
    # Try primary first, fallback if empty
    reply = await call_huggingface_model(user_text, PRIMARY_MODEL)
    if "couldn’t" in reply or "Error" in reply:
        reply = await call_huggingface_model(user_text, FALLBACK_MODEL)
    return reply.strip()

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
        await send_message(chat_id, "Please send text only 🙂")
        return {"ok": True}

    if len(text) > 2000:
        await send_message(chat_id, "Please send shorter messages (under 2000 chars).")
        return {"ok": True}

    reply = await call_ai(text)
    await send_message(chat_id, reply)

    return {"ok": True}
