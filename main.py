import os
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, Request
from pydantic import BaseModel
import httpx

# Load environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not TELEGRAM_TOKEN or not HUGGINGFACE_API_KEY:
    raise RuntimeError("❌ Missing TELEGRAM_TOKEN or HUGGINGFACE_API_KEY")

# ✅ New API endpoint (Inference Providers)
HF_API_URL = "https://router.huggingface.co/hf-inference/facebook/blenderbot-400M-distill"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

app = FastAPI(title="Telegram AI Bot (HuggingFace v2)")

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None


async def call_huggingface(prompt: str) -> str:
    """Call Hugging Face Inference Providers API."""
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": prompt}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):  # retry if model is warming up
            try:
                response = await client.post(HF_API_URL, headers=headers, json=payload)
                if response.status_code == 503:
                    print("🕒 Model warming up...")
                    await asyncio.sleep(5)
                    continue
                response.raise_for_status()
                data = response.json()

                if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()
                elif isinstance(data, dict) and "error" in data:
                    print(f"⚠️ HF API error: {data['error']}")
                    return "The AI model is temporarily busy — please try again later."
                else:
                    return "I’m here! How can I help you today?"
            except Exception as e:
                print(f"❌ HF API call failed: {e}")
                await asyncio.sleep(3)
        return "Sorry — I couldn’t reach the AI service right now."


async def send_message(chat_id: int, text: str):
    """Send a message to Telegram."""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        await client.post(url, json=payload)


@app.post("/webhook")
async def telegram_webhook(update: Message, request: Request):
    """Handle Telegram webhook."""
    data = await request.json()
    message = data.get("message") or data.get("edited_message")
    if not message:
        return {"ok": True, "note": "no message"}

    chat_id = message.get("chat", {}).get("id")
    text = message.get("text") or message.get("caption") or ""
    if not text.strip():
        await send_message(chat_id, "Sorry, I only handle text messages for now.")
        return {"ok": True}

    print(f"📩 From {chat_id}: {text}")
    reply = await call_huggingface(text)
    print(f"🤖 Reply: {reply}")
    await send_message(chat_id, reply)
    return {"ok": True}
