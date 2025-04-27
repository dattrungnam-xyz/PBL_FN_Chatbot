import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
import pinecone

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
load_dotenv()

# Environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME ="ocop-vietnamese768"
PINECONE_NAMESPACE = "ocop_bkai_foundation_model"

# Pinecone init
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Prompt template
prompt = """
Bạn là một trợ lý bán hàng chuyên nghiệp.

Nhiệm vụ của bạn:
- Đọc kỹ yêu cầu của khách hàng.
- Đọc danh sách các sản phẩm có sẵn.
- Chọn ra những sản phẩm phù hợp nhất với nhu cầu của khách hàng.
- Trình bày câu trả lời một cách tự nhiên, lịch sự, hấp dẫn.
- Đưa ra từ 1 đến 3 sản phẩm gợi ý.
- Nếu không tìm thấy sản phẩm phù hợp, hãy xin lỗi khách hàng nhẹ nhàng.

Thông tin khách hàng cung cấp:
"{user_query}"

Danh sách sản phẩm:
{context}

Yêu cầu cách trả lời:
- Viết câu trả lời mạch lạc, tự nhiên, như đang trò chuyện.
- Nếu có thể, đề xuất thêm thông tin nổi bật như nguồn gốc, giá, điểm nổi bật.
- Kết thúc bằng lời mời mua hàng hoặc hỗ trợ thêm.
"""

# ✅ Dùng embedding từ BkAI
embedding_model = HuggingFaceEmbeddings(
    model_name="bkai-foundation-models/vietnamese-bi-encoder",
    model_kwargs={"device": "cpu"},  # hoặc "cuda" nếu bạn có GPU
    encode_kwargs={"normalize_embeddings": True}
)

# Pinecone + LangChain
vector_store = LangchainPinecone(
    index=index,
    embedding=embedding_model,
    text_key="chunk_text",
    namespace=PINECONE_NAMESPACE
)

# LLM Chain
template = PromptTemplate(input_variables=["context", "user_query"], template=prompt)
llm = Ollama(model="gemma:2b", base_url="http://127.0.0.1:11434")
llm_chain = LLMChain(prompt=template, llm=llm)

# API endpoint
@app.route('/chatbot', methods=['POST'])
def chatbot():
    try:
        user_query = request.json.get("query", "")
        print(index.describe_index_stats())
        if not user_query:
            return jsonify({"error": "Missing query"}), 400

        related_docs = vector_store.similarity_search(user_query, k=5)
        context_text = "\n\n".join([doc.page_content for doc in related_docs])
        print("Matched Context:", context_text)

        response = llm_chain.run(context=context_text, user_query=user_query)

        return jsonify({"response": response}), 200

    except Exception as e:
        logger.error(f"Error during processing the query: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    

