import os
import torch
import shutil

# 1. انقل عملية المسح هنا (قبل ما تفتح أي اتصال بالداتابيز)
db_path = "./chroma_db"
if os.path.exists(db_path):
    print(f"🧹 Deleting old database at {db_path}...")
    try:
        shutil.rmtree(db_path)
        print("✅ Clean start: Old database removed.")
    except Exception as e:
        print(f"⚠️ Could not auto-delete: {e}")
        print("👉 Make sure no other script is running!")

# 2. دلوقتي نقدر نعمل Import واحنا مطمنين
from sentence_transformers import SentenceTransformer
from app.rag.vector_db import vector_db
from langchain_text_splitters import RecursiveCharacterTextSplitter 

def ultra_precision_ingestion():
    device ="cpu"
    print(f"🚀 Running on {device} to avoid GPU compatibility issues...")
    
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    
    # تهيئة الكولكشن الجديدة
    vector_db.init_collection() 
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len,
    )

    data_path = "PERFECT_TAFSEER"
    # تأكد إن المسار صح أو استخدم المسار الكامل لو لسه مش شايف الفولدر
    if not os.path.exists(data_path):
        print(f"❌ Error: Folder '{data_path}' not found!")
        return

    files = [f for f in os.listdir(data_path) if f.endswith('.txt')]
    print(f"📂 Processing {len(files)} files with High Precision Chunking...")
    
    chunk_id = 0
    for i, filename in enumerate(files):
        with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
            full_content = f.read().strip()
            if len(full_content) < 15: continue 
            
            chunks = text_splitter.split_text(full_content)
            
            for chunk in chunks:
                embedding = model.encode(chunk, normalize_embeddings=True).tolist()
                
                vector_db.collection.add(
                    ids=[f"{filename}_{chunk_id}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": "high_precision", "file": filename}]
                )
                chunk_id += 1
        
        if i % 10 == 0:
            print(f"🎯 Progress: {i}/{len(files)} files indexed...")

    print(f"⭐ High-Fidelity Database is Ready with {chunk_id} chunks!")

if __name__ == "__main__":
    ultra_precision_ingestion()