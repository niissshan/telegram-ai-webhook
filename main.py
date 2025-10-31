import os
import asyncio
import json
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Strip spaces/newlines from environment values
TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
HUGGINGFACE_API_KEY = (os.getenv("HUGGINGFACE_API_KEY") or "").strip()

if not TELEGRAM_TOKEN or not HUGGINGFACE_API_KEY:
    raise RuntimeError("Set TELEGRAM_TOKEN and HUGGINGFACE_API_KEY in environment")

# URLs
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
HF_API_URL = "https://router.huggingface.co/hf-inference/mistralai/Mistral-7B-Instruct-v0.2"

# FastAPI app
app = FastAPI(title="Telegram AI Webhook")

class Message(BaseModel):
    update_id: int
    message: Dict[str, Any] | None = None
    edited_message: Dict[str, Any] | None = None


# --- Hugging Face AI Call ---
async def call_huggingface(user_text: str) -> str:
    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"inputs": user_text}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(HF_API_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                print(f"❌ HF API call failed: {resp.status_code} - {resp.text}")
                return "Sorry — I couldn’t reach the AI service right now."

            data = resp.json()
            if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
                return data[0]["generated_text"].strip()
            elif isinstance(data, dict) and "generated_text" in data:
                return data["generated_text"].strip()
            else:
                print("❌ Unexpected HF response:", data)
                return "Sorry, I couldn’t generate a proper response."
    except Exception as e:
        print(f"❌ HF API call failed: {e}")
        return "Sorry — I couldn’t reach the AI service right now."


# --- Send Telegram message ---
async def send_message(chat_id: int, text: str):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()


# --- Telegram Webhook ---
@app.post("/webhook")
async def telegram_webhook(update: Message, request: Request):
    data = await request.json()
    message = data.get("message") or data.get("edited_message")

    if not message:
        return {"ok": True, "note": "no message"}

    chat_id = message.get("chat", {}).get("id")
    user_text = message.get("text") or message.get("caption") or ""

    if not user_text.strip():
        await send_message(chat_id, "Sorry, I only handle text messages for now.")
        return {"ok": True}

    if len(user_text) > 2000:
        await send_message(chat_id, "Please send shorter messages (under 2000 characters).")
        return {"ok": True}

    print(f"📩 From {chat_id}: {user_text}")

    # Get AI response
    reply = await call_huggingface(user_text)

    # Send back to Telegram
    try:
        await send_message(chat_id, reply)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed sending message: {e}")

    print(f"🤖 Reply: {reply}")
    return {"ok": True}


# --- Health Check ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "Telegram AI Webhook is running."}


# --- Optional: Test endpoint for HF connection ---
@app.get("/__test_hf")
async def test_hf():
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    async with httpx.AsyncClient() as client:
        r = await client.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": "Hello, how are you?"}
        )
        return {"status": r.status_code, "body": r.text}
