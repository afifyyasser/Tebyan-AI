import os
import re
import asyncio
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import edge_tts
from app.services.agent_brain import TafseerAgent

app = FastAPI()


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# تحميل agent
agent = TafseerAgent()

# نموذج البيانات المحدث لاستقبال الـ mode
class ChatRequest(BaseModel):
    message: str
    mode: str = "text"  # الافتراضي نص

@app.get("/", response_class=HTMLResponse)
async def get_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})

@app.post("/ask")
async def ask_agent(request: ChatRequest):
    try:
        # 1. جلب الإجابة
        answer = agent.work(request.message)
        audio_url = None
        
        if request.mode == "voice":
            # تنظيف النص: حذف كل شيء ماعدا الحروف العربية والمسافات والنقط
            audio_text = re.sub(r'[^\u0600-\u06FF\s.]', '', answer).strip()
            
            # استبدال الفواصل بنقط لضمان الوقفات (النفس)
            audio_text = audio_text.replace('،', '.').replace(':', '.')
            
            # تقصير النص جداً للتجربة (أول 150 حرف) لضمان النجاح السريع
            audio_text = audio_text[:1500]
            
            print(f"🎙️ النص المرسل للمحرك: {audio_text}") # عشان نشوف النص في الـ Terminal

            if len(audio_text) > 5:
                audio_filename = f"res_{uuid.uuid4().hex[:8]}.mp3"
                audio_path = os.path.join("static", audio_filename)
                
                # محاولة توليد الصوت (Try & Retry) بصوتين مختلفين
                success = False
                for voice in ["ar-SA-ShakirNeural", "ar-SA-HamedNeural"]:
                    try:
                        communicate = edge_tts.Communicate(audio_text, voice)
                        await communicate.save(audio_path)
                        success = True
                        break # لو نجح اخرج من اللوب
                    except Exception as inner_e:
                        print(f" فشل الصوت {voice}: {inner_e}")
                
                if success:
                    audio_url = f"/static/{audio_filename}"

        return {"answer": answer, "audio_url": audio_url, "mode": request.mode}

    except Exception as e:
        print(f" خطأ حرج: {e}")
        return {"answer": "حدث خطأ، حاول ثانية.", "audio_url": None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)