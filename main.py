import os
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import requests

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_API_KEY = os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN in environment")

if not HF_API_KEY:
    raise RuntimeError("Missing Hugging Face API key (HF_API_KEY or HUGGINGFACE_API_KEY)")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

app = FastAPI(title="Telegram AI Webhook")

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None


async def call_huggingface(user_text: str) -> str:
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": user_text}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        else:
            return "Sorry, I couldn’t generate a proper response."
    except requests.exceptions.RequestException as e:
        print(f"❌ HuggingFace error: {e}")
        return "Sorry — I couldn’t reach the AI service right now."


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

    chat_id = message.get("chat", {}).get("id")
    text = message.get("text") or message.get("caption") or ""
    if not text.strip():
        await send_message(chat_id, "Please send some text.")
        return {"ok": True}

    if len(text) > 2000:
        await send_message(chat_id, "Please send shorter messages (under 2000 characters).")
        return {"ok": True}

    reply = await call_huggingface(text)
    await send_message(chat_id, reply)
    return {"ok": True}
