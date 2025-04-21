import os
import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_pinecone import Pinecone as LangchainPinecone
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
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
Bạn là một trợ lý ảo chuyên nghiệp, thân thiện, có nhiệm vụ tư vấn và hỗ trợ người dùng về các sản phẩm OCOP của Việt Nam.

Dưới đây là một số thông tin liên quan đến các sản phẩm OCOP:

{context}

Câu hỏi của người dùng: {user_query}

Dựa trên thông tin trên, hãy trả lời một cách ngắn gọn, chính xác và thân thiện. Nếu không có thông tin liên quan, hãy lịch sự cho biết rằng bạn chưa có dữ liệu phù hợp.

Trả lời:
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

        related_docs = vector_store.similarity_search(user_query, k=3)
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
    

