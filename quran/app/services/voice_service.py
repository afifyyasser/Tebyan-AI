import asyncio
import edge_tts
import pygame
import os
from faster_whisper import WhisperModel

class VoiceService:
    def __init__(self):
        # تحميل موديل Faster-Whisper (يعمل على GPU RTX 5060 بسرعة البرق)
        # نستخدم حجم 'small' أو 'medium' للدقة العالية في العربية
        self.stt_model = WhisperModel("small", device="cuda", compute_type="float16")
        pygame.mixer.init()

    def speech_to_text(self, audio_path):
        """تحويل ملف الصوت إلى نص باستخدام Faster-Whisper"""
        segments, info = self.stt_model.transcribe(audio_path, beam_size=5, language="ar")
        text = " ".join([segment.text for segment in segments])
        return text

    async def text_to_speech_edge(self, text, output_path="response.mp3"):
        """تحويل النص لصوت احترافي باستخدام Edge-TTS"""
        # اختيار صوت "شاكر" السعودي أو "فاطمة" ليكون الصوت فصحى وقوي
        voice = "ar-SA-ZariyahNeural" # أو "ar-EG-SalmaNeural" للعامية المصرية
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        
        # تشغيل الصوت
        pygame.mixer.music.load(output_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            await asyncio.sleep(0.1)

    def play_voice(self, text):
        """دالة مساعدة لتشغيل الـ Async داخل الـ UI"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.text_to_speech_edge(text))