import os
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not TELEGRAM_TOKEN or not HUGGINGFACE_API_KEY:
    raise RuntimeError("Set TELEGRAM_TOKEN and HUGGINGFACE_API_KEY in environment")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"

app = FastAPI(title="Telegram AI Webhook")

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None

async def call_huggingface(user_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": f"You are a helpful Telegram assistant. Reply concisely.\nUser: {user_text}\nAssistant:",
        "parameters": {"max_new_tokens": 200, "temperature": 0.6}
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"HuggingFace error {r.status_code}: {r.text}")
        data = r.json()
        # Output format may differ depending on model
        if isinstance(data, list) and len(data) and "generated_text" in data[0]:
            return data[0]["generated_text"].split("Assistant:")[-1].strip()
        return "Sorry, I couldn’t understand that right now."

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

    try:
        reply = await call_huggingface(text)
    except Exception as e:
        await send_message(chat_id, f"AI error: {e}")
        return {"ok": False, "error": str(e)}

    try:
        await send_message(chat_id, reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed sending message: {e}")

    return {"ok": True}
