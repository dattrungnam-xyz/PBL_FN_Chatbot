import os
import logging
import pymysql
import time
import json
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.chains import LLMChain
from langchain_together import Together

# ---------- Load environment ----------
load_dotenv()

# ---------- Flask App ----------
app = Flask(__name__)

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- MySQL Config ----------
MYSQL_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_DATABASE"),
    "port": int(os.getenv("DB_PORT", 3306)),
}

# ---------- Category Mapping ----------
CATEGORY_MAP = {
    "FOOD": "Thực phẩm",
    "BEVERAGE": "Đồ uống",
    "HERB": "Thảo dược",
    "HANDICRAFTS_DECORATION": "Đồ trang trí",
}

# ---------- TF-IDF Embedding Adapter ----------
class TfidfEmbeddings(Embeddings):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []

    def fit(self, docs):
        self.documents = docs
        self.vectorizer.fit(docs)

    def embed_documents(self, texts):
        return self.vectorizer.transform(texts).toarray()

    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0]

# ---------- Load Products ----------
def load_products():
    try:
        conn = pymysql.connect(**MYSQL_CONFIG)
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            query = """
                SELECT 
                    p.id, p.name, p.category, s.province, 
                    p.description, p.price, p.star, p.status
                FROM product p
                JOIN seller s ON p.sellerId = s.id
            """
            cursor.execute(query)
            results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        logger.error(f"MySQL error: {e}")
        return []

# ---------- Build LangChain Components ----------
def build_vectorstore(products):
    docs = []
    metadatas = []
    contents = []

    for p in products:
        content = f"{p['name']} {p['description']} {p['province']} {p['price']}đ {CATEGORY_MAP.get(p['category'], p['category'])} {p['star']} sao OCOP"
        contents.append(content)
        metadata = {"product": p}
        metadatas.append(metadata)
        docs.append(Document(page_content=content, metadata=metadata))

    embedding_model = TfidfEmbeddings()
    embedding_model.fit(contents)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    return vectorstore

# ---------- Prompt Template ----------
prompt_template = PromptTemplate(
    input_variables=["context", "user_query"],
    template="""
Bạn là một trợ lý bán hàng OCOP thân thiện, tận tâm và chuyên nghiệp.

Khách hàng hỏi: "{user_query}"

Dưới đây là danh sách sản phẩm OCOP liên quan nhất:

{context}

Hãy:
- Nếu tìm thấy sản phẩm phù hợp, chọn ra 1–2 sản phẩm nổi bật để tư vấn.
- Với mỗi sản phẩm, hãy giới thiệu tên sản phẩm, tỉnh sản xuất, số sao OCOP, giá, đặc điểm nổi bật.
- Nêu rõ lý do vì sao sản phẩm phù hợp với nhu cầu khách.
- Nếu khách hỏi về việc mua sản phẩm XYZ với số lượng N:
    + Xác định được tên sản phẩm trong danh sách.
    + Tư vấn cửa hàng/công ty sản xuất tương ứng.
    + Tính tổng tiền: giá × số lượng.
    + Trình bày rõ ràng và dễ hiểu.
- Nếu không tìm thấy sản phẩm phù hợp:
    + Trả về id_product là mảng rỗng: []
    + Message là lời xin lỗi nhẹ nhàng và đề xuất khách thử lại sau hoặc gợi ý sản phẩm gần giống.
- Luôn giữ lời văn nhẹ nhàng, dễ đọc, như đang trò chuyện thật sự với khách.
- Không dùng markdown, không dùng tiếng Anh.
- Chỉ trả lời trực tiếp bằng tiếng Việt, không mô tả quá trình phân tích.

Luôn trả về kết quả theo định dạng JSON như sau:
{{
  "id_product": ["<id_sản_phẩm_1>", "<id_sản_phẩm_2>", ...],
  "message": "<Nội dung tư vấn cho khách>"
}}

Kết thúc phần message bằng lời cảm ơn chân thành và lời mời khách quay lại mua hàng.
"""
)

llm = Together(
    model="deepseek-ai/DeepSeek-V3",
    api_key=os.getenv("TOGETHER_API_KEY"),
    temperature=0.7,
    max_tokens=2000,
)

# ---------- Flask Route ----------
@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_query = request.json.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "Missing query"}), 400

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    related_docs = retriever.get_relevant_documents(user_query)

    context = "\n\n".join([
        f"{doc.metadata['product']['name']} | id: {doc.metadata['product']['id']} | Tỉnh: {doc.metadata['product']['province']} | Giá: {doc.metadata['product']['price']}đ | Loại: {CATEGORY_MAP.get(doc.metadata['product']['category'], doc.metadata['product']['category'])} | OCOP: {doc.metadata['product']['star']} sao\nMô tả: {doc.metadata['product']['description']}"
        for doc in related_docs
    ])

    chain = LLMChain(llm=llm, prompt=prompt_template)
    result = chain.run({"context": context, "user_query": user_query})

    try:
        json_str = re.search(r"\{.*\}", result, re.DOTALL).group()
        parsed = json.loads(json_str)
    except Exception as e:
        return jsonify({"error": "LLM parsing error", "raw": result}), 500

    return jsonify({
        "id_product": parsed.get("id_product", []),
        "message": parsed.get("message", "")
    })

# ---------- Init Data ----------
logger.info("⏳ Loading products and building vectorstore...")
products = load_products()
if not products:
    logger.error("❌ No products found. Exiting.")
    exit(1)

vectorstore = build_vectorstore(products)
logger.info("✅ Vectorstore is ready.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
