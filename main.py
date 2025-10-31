import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
HF_API_KEY = os.getenv("HF_API_KEY")

# Hugging Face model (small free model)
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Telegram API URL
TELEGRAM_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def get_ai_response(prompt):
    """
    Get AI-generated response from Hugging Face or fallback model.
    """
    # 1️⃣ Try Hugging Face with API key (if provided)
    if HF_API_KEY:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        payload = {"inputs": prompt}
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list) and "generated_text" in data[0]:
                return data[0]["generated_text"]
            elif isinstance(data, dict) and "error" in data:
                print("❌ HuggingFace error:", data["error"])
        else:
            print(f"❌ HuggingFace error: {response.status_code} - {response.text}")

    # 2️⃣ Fallback: use a completely free local response
    try:
        return simple_local_ai(prompt)
    except Exception as e:
        print("⚠️ Local AI fallback failed:", str(e))
        return "Sorry, I couldn’t generate a proper response right now."


def simple_local_ai(prompt):
    """
    Very simple fallback model (no API key needed).
    """
    prompt = prompt.lower()
    if "hi" in prompt or "hello" in prompt:
        return "Hey there 👋! How can I help you today?"
    elif "how are you" in prompt:
        return "I'm just code, but I'm feeling great when you chat with me 😄"
    elif "your name" in prompt:
        return "I'm your friendly Telegram AI bot 🤖 built with FastAPI!"
    elif "bye" in prompt:
        return "Goodbye 👋, talk to you soon!"
    else:
        return "I'm here! You can ask me anything 😊"


def send_message(chat_id, text):
    """
    Send a message back to the Telegram user.
    """
    url = f"{TELEGRAM_URL}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    requests.post(url, json=payload)


@app.post("/webhook")
async def webhook(request: Request):
    """
    Receive updates from Telegram.
    """
    data = await request.json()
    if "message" in data and "text" in data["message"]:
        chat_id = data["message"]["chat"]["id"]
        user_message = data["message"]["text"]

        print(f"📩 Received: {user_message}")

        bot_response = get_ai_response(user_message)
        send_message(chat_id, bot_response)

    return JSONResponse(content={"ok": True})


@app.get("/")
def home():
    return {"message": "Telegram AI Bot is live 🚀"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
