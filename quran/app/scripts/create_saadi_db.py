import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- الإعدادات ---
PDF_PATH = "data/tafseer_al_saadi.pdf"  # المسار اللي اتفقنا عليه
SAVE_PATH = "vector_db/saadi_index"      # اسم ومسار الـ VD الثانية
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def build_saadi_vector_store():
    # 1. التأكد من وجود المجلدات المطلوبة
    if not os.path.exists("vector_db"):
        os.makedirs("vector_db")
        print("Created 'vector_db' directory.")

    # 2. تحميل الكتاب (نصي)
    print(f"Loading PDF from: {PDF_PATH}...")
    loader = PyMuPDFLoader(PDF_PATH)
    docs = loader.load()

    # 3. تقسيم النص (Chunking) بطريقة تناسب التفسير
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)

    # 4. تجهيز موديل الـ Embedding
    print("Initializing Embedding Model (Multilingual)...")
    embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={'device': 'cpu'} # إجبار الموديل يشتغل على البروسيسور
)

    # 5. إنشاء الـ Vector Store وحفظه
    print("Building FAISS index... this might take a minute.")
    vector_store = FAISS.from_documents(splits, embeddings)
    
    vector_store.save_local(SAVE_PATH)
    print(f"Done! Vector Database saved successfully at: {SAVE_PATH}")

if __name__ == "__main__":
    if os.path.exists(PDF_PATH):
        build_saadi_vector_store()
    else:
        print(f"Error: Could not find the PDF file at {PDF_PATH}. Please check the 'data' folder.")