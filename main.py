import os
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from openai import OpenAI  # ✅ New import for OpenAI v2

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Set TELEGRAM_TOKEN and OPENAI_API_KEY in environment")

# ✅ Create OpenAI client (v2 style)
client = OpenAI(api_key=OPENAI_API_KEY)

TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
app = FastAPI(title="Telegram AI Webhook")

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None

async def call_openai(user_text: str) -> str:
    def sync_call():
        # ✅ Updated syntax for new API
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a friendly, concise assistant that helps users on Telegram."},
                {"role": "user", "content": user_text}
            ],
            max_tokens=500,
            temperature=0.6,
        )
        return resp.choices[0].message.content.strip()
    return await asyncio.to_thread(sync_call)

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
        reply = await call_openai(text)
    except Exception as e:
        print("❌ Error calling OpenAI:", e)  # ✅ Log error in Render logs
        await send_message(chat_id, "Sorry — I couldn't reach the AI service right now.")
        return {"ok": False, "error": str(e)}

    try:
        await send_message(chat_id, reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed sending message: {e}")

    return {"ok": True}
