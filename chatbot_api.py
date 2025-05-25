import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from recommend import shared_data, data_lock, tfidf_vectorizer  # Nếu cùng process
from flask import request, jsonify
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_together import Together
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LangChain setup
prompt_template = PromptTemplate(
    input_variables=["context", "user_query"],
    template="""
Bạn là một trợ lý tư vấn bán hàng chuyên nghiệp.

Nhu cầu khách hàng:
"{user_query}"

Danh sách sản phẩm phù hợp:
{context}

Viết câu trả lời thân thiện, chuyên nghiệp và tự nhiên.
"""
)

llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    api_key=os.getenv("TOGETHER_API_KEY"),
    temperature=0.7,
)

llm_chain = LLMChain(prompt=prompt_template, llm=llm)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        user_query = request.json.get("query", "")
        if not user_query:
            return jsonify({"error": "Missing query"}), 400

        with data_lock:
            data_snapshot = shared_data

        if not data_snapshot:
            return jsonify({"error": "Product data not ready"}), 503

        # TF-IDF vectorize query
        user_vector = tfidf_vectorizer.transform([user_query])
        similarities = cosine_similarity(user_vector, data_snapshot.tfidf_matrix).flatten()

        # Lấy top 5 sản phẩm gần nhất
        top_indices = similarities.argsort()[-5:][::-1]
        top_products = [data_snapshot.products[i] for i in top_indices]

        # Tạo context text
        context_text = "\n\n".join([
            f"{p['name']} ({p['province']}) - {p['price']}đ - {p['description']}" for p in top_products
        ])

        # Sinh câu trả lời từ LLM
        response = llm_chain.run(context=context_text, user_query=user_query)

        return jsonify({"response": response}), 200

    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({"error": str(e)}), 500
