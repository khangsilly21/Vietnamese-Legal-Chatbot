from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
import requests
from resourses import innit_app, create_chat_engine
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.chatbot ={}
    app.state.chatbot['query_engine'], app.state.chatbot['custom_prompt'], app.state.chatbot['custom_chat_history'], app.state.chatbot['llm'] = (result := innit_app()).values()
    yield
   
app = FastAPI(lifespan=lifespan)

import os
from dotenv import load_dotenv
load_dotenv()

VERIFY_TOKEN = os.getenv('FB_VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "your_page_access_token")




@app.get("/")
async def root():
    return {"message": "Messenger Bot is running!"}

@app.get('/webhook')
def init_messenger(request: Request):

    fb_token = request.query_params.get("hub.verify_token")

    # we verify if the token sent matches our verify token
    if fb_token == VERIFY_TOKEN:
    	# respond with hub.challenge parameter from the request.
        return Response(content=request.query_params["hub.challenge"])
    return 'Failed to verify token'

@app.post("/webhook")
async def receive_message(request: Request):
    data = await request.json()
    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            sender_id = event["sender"]["id"]
            if "message" in event:
                text = event["message"].get("text", "")
                if text:
                    chatbot_engine = create_chat_engine(
                        app.state.chatbot['query_engine'],
                        app.state.chatbot['custom_prompt'],
                        app.state.chatbot['custom_chat_history'],
                        app.state.chatbot['llm']
                    )
                    response_chat = chatbot_engine.chat(text)
                    try:
                        reply = response_chat.response.strip()
                        send_message(sender_id, reply)
                    except Exception as e:
                        print("Error:", e)
                        send_message(sender_id, "Xin lỗi, tôi không thể trả lời câu hỏi của bạn ngay bây giờ.")                  
    return {"status": "ok"}

def send_message(recipient_id, message_text):
    url = "https://graph.facebook.com/v19.0/me/messages"
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": message_text},
        "messaging_type": "RESPONSE"
    }
    params = {"access_token": PAGE_ACCESS_TOKEN}
    response = requests.post(url, params=params, json=payload)
    if response.status_code != 200:
        print("Error sending message:", response.text)
