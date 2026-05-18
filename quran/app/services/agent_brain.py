import os
import sys
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. تحميل متغيرات البيئة وإعداد المسارات
load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.services.rag_service import rag_handler

class TafseerAgent:
    def __init__(self):
        # إعداد عميل OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.rag = rag_handler
        
        # --- نظام الذاكرة (للاحتفاظ بسياق الحوار) ---
        self.chat_history = [] 
        self.max_history_length = 6 # الاحتفاظ بآخر 3 أسئلة و3 إجابات
        
        # 2. إعداد قاعدة بيانات كتب التوحيد (ChromaDB)
        db_path = os.path.join(os.getcwd(), "chroma_db")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        try:
            self.collection = self.chroma_client.get_collection(name="tafseer_collection")
        except Exception as e:
            print(f"⚠️ تحذير: لم يتم العثور على chroma collection: {e}")

        # 3. إعداد قاعدة بيانات تفسير السعدي (FAISS)
        # نستخدم الـ CPU لضمان استقرار العمل مع كرت RTX 5060
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        saadi_db_path = os.path.join(os.getcwd(), "vector_db", "saadi_index")
        
        if os.path.exists(saadi_db_path):
            self.saadi_vector_db = FAISS.load_local(
                saadi_db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print("✅ تم تحميل قاعدة بيانات تفسير السعدي بنجاح.")
        else:
            print("⚠️ تحذير: مسار FAISS غير موجود!")

    def work(self, user_query):
        # --- المرحلة الأولى: البحث المزدوج في المصادر ---
        
        # البحث في كتب التوحيد
        raw_docs_touhid = self.rag.hybrid_search_with_rrf(user_query, self.collection, top_k=15)
        
        # البحث في تفسير السعدي
        saadi_results = self.saadi_vector_db.similarity_search(user_query, k=5)
        raw_docs_saadi = [doc.page_content for doc in saadi_results]

        # دمج وإعادة ترتيب النتائج (Reranking)
        combined_docs = raw_docs_touhid + raw_docs_saadi
        final_docs = self.rag.rerank(user_query, combined_docs)
        context = "\n\n".join(final_docs[:7])

        # --- المرحلة الثانية: بناء الرسائل وتاريخ الحوار ---
        
        messages = [
            {
                "role": "system", 
                "content": "أنت عالم متخصص يجمع بين التفسير والعقيدة. مهمتك دمج المصادر لتقديم إجابة شرعية متكاملة وموثقة."
            }
        ]
        
        # حقن الذاكرة (تاريخ الحوار السابق)
        messages.extend(self.chat_history)
        
        # --- المرحلة الثالثة: البرومبت الاحترافي مع القواعد الصارمة ---
        full_prompt = f"""
        ممنوع تماماً تبدأ إجابتك بالسلام أو التحية (مثل: وعليكم السلام، أهلاً بك، حياك الله) الا لو لقيت تحيه. ابدأ بالرد على السؤال مباشرة وبشكل مركز بناءً على السياق المتاح لك."

### أسلوب التفاعل (مهم جداً):
- تصرف كإنسان طبيعي في الحوار، وليس كآلة.
- إذا بدأ المستخدم بتحية (مثل: السلام عليكم، ازيك، hello)، يجب أن ترد التحية أولاً بشكل طبيعي قبل أي إجابة.
- لا تبدأ مباشرة في إعطاء إجابة علمية إلا إذا السؤال واضح ومباشر.
- إذا كان الكلام عادي أو دردشة، رد بشكل بسيط وطبيعي.
- استخدم لغة سهلة وبشرية، وتجنب الأسلوب الرسمي المبالغ فيه.
- يمكنك إضافة جملة تمهيدية طبيعية قبل الإجابة (مثل: تمام، خلينا نشوف...).

---

### سياق العمل:
لديك وصول لمصادر متعددة (كتب التوحيد + تفسير السعدي). 

### النصوص المرجعية المستخرجة من الكتب:
{context}

### السؤال الحالي المطلوب الإجابة عليه:
{user_query}

### القواعد الصارمة للمعالجة (Instructions):
1. **التكامل:** إذا كان السؤال عن آية قرآنيّة، ابدأ بذكر ما قاله السعدي في التفسير أولاً، ثم اربطه بالأحكام العقدية من كتب التوحيد.
2. **الدقة:** ميز بدقة متناهية بين أنواع الأحكام (عقدية، فقهية، سلوكية).
3. **التوثيق:** يجب الإشارة بوضوح في ثنايا الإجابة إذا كان الكلام مستنداً إلى "تفسير السعدي" أو "كتب التوحيد".
4. **الذاكرة:** إذا كان السؤال الحالي مرتبطاً بأسئلة سابقة في الحوار، استخدم الذاكرة المتاحة لتقديم إجابة متسلسلة ومنطقية.
5. **الأمانة العلمية:** لا تؤلف معلومات من عندك، التزم بما ورد في النصوص المرفقة فقط.
6. **ابحث في المصادر بدقه 
7. **عدم التفكرير او الاجابه من اي مصدر اخر 

---

### طريقة بناء الرد:
1. ابدأ برد بشري طبيعي (تحية أو جملة تمهيدية حسب السياق).
2. ثم قدم الإجابة العلمية المنظمة بالشكل التالي:

- **الحكم المستخلص:** (اكتب الحكم الشرعي المستخلص بدقة)
- **من تفسير الآيات (إن وجد):** (اقتبس من السعدي ما يوضح المعنى)
- **الدليل والتعليل العقدي:** (شرح مفصل يربط الدليل بالمسألة العقدية)

---

الإجابة الموثقة:
"""
        # إضافة البرومبت النهائي كرسالة مستخدم
        messages.append({"role": "user", "content": full_prompt})

        # --- المرحلة الرابعة: استدعاء GPT-4o وتحديث الذاكرة ---
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1 
        )
        
        answer = response.choices[0].message.content

        # حفظ السؤال والإجابة في الذاكرة للجولة القادمة
        self.chat_history.append({"role": "user", "content": user_query})
        self.chat_history.append({"role": "assistant", "content": answer})
        
        # تنظيف الذاكرة القديمة (Trimming)
        if len(self.chat_history) > self.max_history_length:
            self.chat_history = self.chat_history[-self.max_history_length:]
            
        return answer