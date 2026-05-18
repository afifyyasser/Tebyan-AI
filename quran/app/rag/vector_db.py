import chromadb
import os

class VectorDB:
    def __init__(self):
        # تحديد مكان حفظ البيانات (فولدر اسمه chroma_db)
        self.db_path = os.path.join(os.getcwd(), "chroma_db")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection_name = "tafseer_collection"

    def init_collection(self):
        try:
            # إنشاء الـ Collection أو تحميلها لو موجودة
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            print(f"✅ ChromaDB Collection '{self.collection_name}' is ready.")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")

# Instance للاستخدام في باقي الملفات
vector_db = VectorDB()