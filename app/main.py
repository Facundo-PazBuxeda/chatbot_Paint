from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from chat_bot import PaintChatbot
from core.config import settings
import logging
import traceback
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Paint Chatbot API")
bot = PaintChatbot()

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
async def read_root():
    return FileResponse("index.html")

class WhatsAppMessage(BaseModel):
    from_number: str
    message: str

# Your existing endpoints
@app.post("/whatsapp/webhook", tags=["WhatsApp"])
async def whatsapp_webhook(message: WhatsAppMessage):
    try:
        logger.info(f"Processing WhatsApp message from {message.from_number}")
        response = bot.process_message(message.from_number, message.message)
        return {"status": "success", "message": response}
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"Error in WhatsApp webhook: {str(e)}\nTraceback:\n{trace}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{user_id}/history", tags=["Test"])
async def get_chat_history(user_id: str):
    try:
        history = bot.get_chat_history(user_id)
        if not history:
            raise HTTPException(status_code=404, detail="No history found")
        return {"history": history}
    except HTTPException:
        raise
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(
            f"Error fetching history for {user_id}: {str(e)}\nTraceback:\n{trace}"
        )
        raise HTTPException(status_code=500, detail=str(e))
