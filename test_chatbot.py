import os
import logging
import pymysql
import time
import numpy as np
import json
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_together import Together

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

# ---------- Shared Data ----------
class DataSnapshot:
    def __init__(self, products, tfidf_matrix):
        self.products = products
        self.tfidf_matrix = tfidf_matrix

shared_data = None
tfidf_vectorizer = None

# ---------- Category Mapping ----------
CATEGORY_MAP = {
    "FOOD": "Thực phẩm",
    "BEVERAGE": "Đồ uống",
    "HERB": "Thảo dược",
    "HANDICRAFTS_DECORATION": "Đồ trang trí",
}

# ---------- Load Products from MySQL ----------
def load_products_from_mysql():
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            query = """
                SELECT 
                    p.id, 
                    p.name, 
                    p.category, 
                    s.province, 
                    p.description, 
                    p.price, 
                    p.star,
                    p.status
                FROM product p
                JOIN seller s ON p.sellerId = s.id
           
            """
            cursor.execute(query)
            results = cursor.fetchall()
           
        connection.close()
        logger.info("✅ Loaded products from MySQL")
        
        return results
    except Exception as e:
        logger.error(f"⚠️ MySQL error: {e}")
        return []

# ---------- TF-IDF Embedding ----------
def load_products_and_vectorize():
    global shared_data, tfidf_vectorizer

    products = load_products_from_mysql()
    if not products:
        logger.warning("⚠️ No products loaded")
        return

    documents = []
    for p in products:
        doc = f"{p['name']} {p['description']} {p['province']} {p['price']}đ {CATEGORY_MAP.get(p['category'], p['category'])} {p['star']} sao OCOP"
        documents.append(doc)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    shared_data = DataSnapshot(products, tfidf_matrix)
    logger.info(f"✅ Vectorized {len(products)} products")
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

llm_chain = LLMChain(prompt=prompt_template, llm=llm)

# ---------- Chatbot API ----------
@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        user_query = request.json.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "Missing query"}), 400

        if not shared_data or not tfidf_vectorizer:
            return jsonify({"error": "Product data not ready"}), 503

        user_vector = tfidf_vectorizer.transform([user_query])
        similarities = cosine_similarity(user_vector, shared_data.tfidf_matrix).flatten()

        top_indices = similarities.argsort()[-5:][::-1]
        top_products = [shared_data.products[i] for i in top_indices]
        print(top_products)
        context_text = "\n\n".join([
            f"{p['name']} |  id: {p['id']} | Tỉnh: {p['province']} | Giá: {p['price']}đ | Loại: {CATEGORY_MAP.get(p['category'], p['category'])} | OCOP: {p['star']} sao\nMô tả: {p['description']}"
            for p in top_products
        ])
        import json
        import re

        response = llm_chain.run(context=context_text, user_query=user_query)

        try:
            json_str = re.search(r"\{.*\}", response, re.DOTALL).group()
            parsed_response = json.loads(json_str)
        except Exception as e:
            print("Lỗi khi trích xuất hoặc parse JSON:", e)
            print("Response nhận được từ LLM:", response)
            parsed_response = {}

        # Bây giờ bạn có thể truy cập như dict
        print(parsed_response["id_product"])
        print(parsed_response["message"])
        # Trả về đúng định dạng
        return jsonify({
            "id_product": parsed_response["id_product"],
            "message": parsed_response["message"]
        }), 200

    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------- Init on Start ----------
def load_and_wait():
    load_products_and_vectorize()
    for _ in range(10):
        if shared_data:
            break
        logger.info("⏳ Waiting for data to be ready...")
        time.sleep(1)
    else:
        logger.error("❌ Timeout: Failed to load product data in time.")

if __name__ == "__main__":
    load_and_wait()
    if shared_data is None:
        logger.error("❌ Không thể khởi động vì chưa có dữ liệu sản phẩm.")
    else:
        app.run(host="0.0.0.0", port=5000)
