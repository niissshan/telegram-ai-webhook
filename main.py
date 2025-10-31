import os
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import httpx

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not TELEGRAM_TOKEN or not HUGGINGFACE_API_KEY:
    raise RuntimeError("❌ Missing TELEGRAM_TOKEN or HUGGINGFACE_API_KEY in environment variables")

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI(title="Telegram AI Bot")

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None

async def call_huggingface(prompt: str) -> str:
    """Send user input to Hugging Face model and return AI reply."""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": prompt}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Try up to 3 times (useful for model warm-up)
            for attempt in range(3):
                response = await client.post(HF_API_URL, headers=headers, json=payload)
                if response.status_code == 503:  # model still loading
                    print("🕒 Model is warming up, retrying...")
                    await asyncio.sleep(5)
                    continue
                response.raise_for_status()
                data = response.json()
                if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()
                elif isinstance(data, dict) and "error" in data:
                    print(f"⚠️ HF API error: {data['error']}")
                    return "The AI model is temporarily busy — try again in a few seconds."
                else:
                    return "I'm here! How can I help you today?"
            return "Sorry — the AI model is taking too long to respond."
        except Exception as e:
            print(f"❌ HuggingFace API call failed: {e}")
            return "Sorry — I couldn’t reach the AI service right now."

async def send_message(chat_id: int, text: str):
    """Send a message back to the user on Telegram."""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

@app.post("/webhook")
async def telegram_webhook(update: Message, request: Request):
    """Handle incoming Telegram messages."""
    data = await request.json()
    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"ok": True, "note": "no message"}

    chat_id = message.get("chat", {}).get("id")
    text = message.get("text") or message.get("caption") or ""
    if not text.strip():
        await send_message(chat_id, "Sorry, I only handle text messages for now.")
        return {"ok": True}

    print(f"📩 Message from {chat_id}: {text}")
    reply = await call_huggingface(text)
    print(f"🤖 Reply: {reply}")
    await send_message(chat_id, reply)
    return {"ok": True}
