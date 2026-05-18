# Tebyan-AI
"A fully local, high-precision Multimodal RAG platform specializing in classical Islamic Tafsir. Powered by Faster-Whisper (STT) on CUDA, BGE-M3 Embeddings with Cross-Encoder Reranking, and Edge-TTS."
🚀 Local Installation & Setup
1. Prerequisites
Python 3.10+ environment.

Properly configured GPU drivers and CUDA Toolkit matching your hardware setup to execute Faster-Whisper using float16 acceleration.

2. Dependency Installation
Open your terminal inside the project directory and run:

Bash
pip install -r requirements.txt
Note: Ensure your requirements.txt file includes the core packages:
fastapi, uvicorn, faster-whisper, edge-tts, pygame, pydantic, jinja2, chromadb (or your chosen vector database provider).

3. Prompt Configuration
The app/services/agent_brain.py file exposes the full_prompt variable, which strictly enforces the agent's behavior, operational rules, and structured architecture outputs.

4. Running the Application
Launch the local development server using:

Bash
python api.py
Once initialized, navigate to your web browser and open: http://127.0.0.1:8000

💡 Engineered Solutions to Technical Challenges
The Greeting Repetition Loop: Addressed via custom System Prompt constraints preventing the model from initiating responses with repetitive salutations unless directly greeted first, backed by a lightweight server-side regex filter.

Dynamic Mode Switching: Built an intelligent pipeline that automatically routes input based on the modality; text inputs trigger optimized text-only streaming to preserve resources, while microphone inputs trigger the complete (STT -> RAG -> TTS) pipeline to speak back to the user.

Acoustic Optimization (TTS Enhancement): Configured speech pacing dynamically (rate="+20%") to eliminate mechanical monotony, combined with punctuation injection (...) to simulate natural breathing intervals.

📝 Contribution & Licensing
This platform is developed to bridge classical Islamic heritage with modern artificial intelligence systems through advanced retrieval-augmented pipelines. Contributions extending the library's scope to include classical Hadith and Fiqh texts are highly welcome.

Developed by: Afify Yasser 🚀
"""

with open("README_en.md", "w", encoding="utf-8") as f:
f.write(markdown_content_en)
print("File README_en.md generated successfully.")

تمام يا هندسة، عدلت لك الملف بالكامل للغة الإنجليزية بناءً على طلبك، وحدثت اسم المطور في نهاية الملف ليكون **Afify Yasser** بالشكل المظبوط والمحترف جداً عشان ترفعه على GitHub.

ملف الـ Markdown باللغة الإنجليزية جاهز الآن للتحميل:
[file-tag: code-generated-file-0-1779132169749931420]

إليك المعاينة الكاملة للنص الإنجليزي للتأكد منه قبل الاستخدام:

```markdown
# 📖 Tebyan AI (تبيان) - Advanced Multimodal RAG Platform

**Tebyan AI** is a cutting-edge, fully local system powered by Generative AI designed for high-precision information retrieval and search across classical Islamic jurisprudence and exegesis (Tafsir). Built on a robust **Dual Vector Store RAG** architecture, the platform integrates state-of-the-art voice processing (Multimodal Interaction) to deliver a seamless, intelligent user experience.

The system is currently specialized for cross-referenced, high-accuracy lookup in specific foundational sources: **Tafsir al-Sa'di** for linguistic clarity and narrative context, and the extensive commentaries of **Sheikh Ibn Uthaymeen** (comprising over **14,000 documents/records**) for deep theological and grammatical analysis.

---

## 🛠️ Tech Stack & Architecture

The system architecture is engineered to run entirely on local hardware, maximizing the performance of modern GPUs (e.g., **NVIDIA RTX 50-Series / 40-Series**):

* **Dual Vector Store Architecture:** Data is isolated into two completely distinct vector databases to prevent contextual bleed and ensure clear separation between the respective scientific methodologies (one repository for Tafsir al-Sa'di, and a separate, independent repository for Ibn Uthaymeen).
* **Semantic Embeddings:** Powered by the **BGE-M3** multilingual model, chosen for its exceptional capability in grasping complex Arabic jurisprudence terms and understanding classical Arabic text structures.
* **Advanced Retrieval & Reranking:** To reduce hallucinations to absolute zero, a **BGE-Reranker (Cross-Encoder)** pipeline is integrated. It re-evaluates and re-orders the retrieved chunks, delivering only the most contextually relevant document to the Large Language Model (LLM).
* **Speech-to-Text (STT):** Powered locally by **Faster-Whisper** (Small/Medium variants) running directly on the GPU via **CUDA** with **float16** precision, achieving near-zero latency text transcription.
* **Text-to-Speech (TTS):** Utilizes the **Edge-TTS** engine (utilizing *Shakir* and *Hamed* neural voices) paired with a custom algorithmic text-processing layer that injects human-like pauses and optimizes speech pacing to mirror the dignity of classical scholarship.
* **Backend Framework:** Built natively using **FastAPI** to utilize full asynchronous execution and high-throughput endpoint routing.

---

## ⚙️ Project Structure

```text
Tebyan_AI/
├── app/
│   ├── services/
│   │   ├── agent_brain.py       # Manages System Prompt and Agent Personality (full_prompt)
│   │   └── voice_service.py     # Voice processing pipeline via Faster-Whisper & Edge-TTS
├── static/                      # Cache for temporary audio files, logs, and assets
├── templates/
│   └── index.html               # Responsive Frontend UI supporting multimodal interaction Modes
├── api.py                       # Core FastAPI application initialization and endpoints
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation (This file)
🚀 Local Installation & Setup
1. Prerequisites
Python 3.10+ environment.

Properly configured GPU drivers and CUDA Toolkit matching your hardware setup to execute Faster-Whisper using float16 acceleration.

2. Dependency Installation
Open your terminal inside the project directory and run:

Bash
pip install -r requirements.txt
Note: Ensure your requirements.txt file includes the core packages:
fastapi, uvicorn, faster-whisper, edge-tts, pygame, pydantic, jinja2, chromadb (or your chosen vector database provider).

3. Prompt Configuration
The app/services/agent_brain.py file exposes the full_prompt variable, which strictly enforces the agent's behavior, operational rules, and structured architecture outputs.

4. Running the Application
Launch the local development server using:

Bash
python api.py
Once initialized, navigate to your web browser and open: http://127.0.0.1:8000

💡 Engineered Solutions to Technical Challenges
The Greeting Repetition Loop: Addressed via custom System Prompt constraints preventing the model from initiating responses with repetitive salutations unless directly greeted first, backed by a lightweight server-side regex filter.

Dynamic Mode Switching: Built an intelligent pipeline that automatically routes input based on the modality; text inputs trigger optimized text-only streaming to preserve resources, while microphone inputs trigger the complete (STT -> RAG -> TTS) pipeline to speak back to the user.

Acoustic Optimization (TTS Enhancement): Configured speech pacing dynamically (rate="+20%") to eliminate mechanical monotony, combined with punctuation injection (...) to simulate natural breathing intervals.

📝 Contribution & Licensing
This platform is developed to bridge classical Islamic heritage with modern artificial intelligence systems through advanced retrieval-augmented pipelines. Contributions extending the library's scope to include classical Hadith and Fiqh texts are highly welcome.

Developed by: Afify Yasser 🚀
