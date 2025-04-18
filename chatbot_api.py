import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama  # Dùng LLM local từ Ollama
from langchain_pinecone import Pinecone as LangchainPinecone  # Cập nhật theo phiên bản mới của LangChain
from langchain_huggingface import HuggingFaceEmbeddings  # Sử dụng gói mới của HuggingFaceEmbeddings
import pinecone

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Khởi tạo Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

# Lấy các biến từ environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ocop-vietnamese")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ocop")  # Namespace của bạn

# Khởi tạo đối tượng Pinecone (thay vì pinecone.init())
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Tạo kết nối với chỉ mục Pinecone và namespace
index = pc.Index(PINECONE_INDEX_NAME)

# Tạo PromptTemplate cho chatbot
prompt = """
Bạn là một trợ lý ảo thân thiện, thông minh, chuyên hỗ trợ người dùng tra cứu sản phẩm OCOP (Chương trình Mỗi xã một sản phẩm Việt Nam).

Bạn có thể:
- Trả lời câu hỏi về sản phẩm như: tên, mô tả, giá, nguồn gốc, cửa hàng bán, số sao OCOP.
- Gợi ý sản phẩm phù hợp với nhu cầu người dùng.
- Phản hồi thân thiện với các câu giao tiếp cơ bản như: "xin chào", "bạn là ai", "cảm ơn", "tôi cần giúp đỡ", v.v.
- Nếu không có thông tin phù hợp, hãy trả lời lịch sự rằng chưa có dữ liệu.

Hãy trả lời thân thiện, dễ hiểu, ngắn gọn, đúng thông tin nhất có thể.

---

Dưới đây là dữ liệu các sản phẩm OCOP có liên quan:

{context}

---

Câu hỏi người dùng: {user_query}

Trả lời:
"""


# Khởi tạo mô hình embedding từ HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# Kết nối Pinecone với Langchain, thêm namespace vào vector store
vector_store = LangchainPinecone(index=index, embedding=embedding_model, text_key="chunk_text", namespace=PINECONE_NAMESPACE)

# Tạo LLM Chain với Ollama (dùng model local)
template = PromptTemplate(input_variables=["context", "user_query"], template=prompt)
llm = Ollama(model="gemma:2b", base_url="http://127.0.0.1:11434")  # Bạn có thể thay bằng "llama3", "gemma", v.v.
llm_chain = LLMChain(prompt=template, llm=llm)

# Định nghĩa API route cho chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        # Lấy câu hỏi từ người dùng
        user_query = request.json.get("query", "")

        if not user_query:
            return jsonify({"error": "Missing query"}), 400

        # Truy vấn vector tương tự từ Pinecone với namespace
        related_docs = vector_store.similarity_search(user_query, k=3)
        # print(related_docs)

        # Gộp nội dung các tài liệu tìm được
        context_text = "\n\n".join([doc.page_content for doc in related_docs])
        print(context_text)

        # Gọi LLM Chain để trả lời câu hỏi dựa trên context
        response = llm_chain.run(context=context_text, user_query=user_query)

        # Trả về kết quả
        return jsonify({"response": response}), 200

    except Exception as e:
        logger.error(f"Error during processing the query: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Chạy Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
