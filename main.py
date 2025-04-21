import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import os
import logging
import sys

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment():
    load_dotenv()
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = "ocop-vietnamese768"

    if not api_key:
        logger.error("Không tìm thấy PINECONE_API_KEY")
        sys.exit(1)

    return api_key, index_name

def initialize_pinecone(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)

    if not pc.has_index(index_name):
        logger.info(f"Tạo mới Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    return pc.Index(index_name)

def process_and_embed_data(df, splitter, model):
    vectors = []
    id_counter = 1

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Xử lý sản phẩm"):
        name = row.get("product_name", "")
        category = row.get("category", "")
        origin = row.get("location", "")
        store = row.get("store", "")
        rating = row.get("star", 0)
        price = row.get("price", "")  # Nếu có cột giá
        description = str(row.get("description", ""))

        if not description.strip():
            logger.warning(f"Bỏ qua sản phẩm rỗng mô tả: {name}")
            continue

        # ✅ Gộp các thông tin thành văn bản đầy đủ
        full_text = f"""
Tên sản phẩm: {name}
Loại sản phẩm: {category}
Nguồn gốc: {origin}
Cửa hàng: {store}
Giá: {price}
Số sao OCOP: {rating}
Mô tả: {description}
""".strip()

        chunks = splitter.split_text(full_text)
        embeddings = model.encode(chunks)

        for i, embedding in enumerate(embeddings):
            vector = {
                "id": f"vec-{id_counter}",
                "values": embedding.tolist(),
                "metadata": {
                    "tên sản phẩm": name,
                    "category": category,
                    "nguồn gốc": origin,
                    "store": store,
                    "số sao": rating,
                    "chunk_index": i,
                    "chunk_text": chunks[i]
                }
            }
            vectors.append(vector)
            id_counter += 1

    return vectors

def main():
    try:
        # Load biến môi trường
        api_key, index_name = load_environment()

        # Khởi tạo Pinecone
        index = initialize_pinecone(api_key, index_name)

        # Tải dữ liệu sản phẩm
        logger.info("Đang tải dữ liệu sản phẩm...")
        df = pd.read_csv("product.csv")
        logger.info(f"Tổng cộng {len(df)} sản phẩm.")

        # Mô hình embedding
        model = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")

        # Text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
        )

        # Tạo vectors
        vectors = process_and_embed_data(df, splitter, model)
        logger.info(f"Tổng cộng {len(vectors)} vectors sẽ được upsert.")

        # Upsert thủ công
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch, namespace="ocop_bkai_foundation_model")
            logger.info(f"Đã upsert {len(batch)} vectors")

        logger.info("Hoàn tất upsert vào Pinecone.")

    except Exception as e:
        logger.error(f"Lỗi: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()