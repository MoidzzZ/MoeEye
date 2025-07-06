import os
import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from huggingface_hub import snapshot_download
import tqdm  # 使用tqdm来显示进度条

DATA_DIR = "./data"
DB_PATH = "./chroma_db_prod"  # 持久化数据库的存储路径
COLLECTION_NAME = "character_references"
MODEL_ID = "openai/clip-vit-base-patch32"

# --- 1. 初始化模型和ChromaDB ---
device = "cuda" if torch.cuda.is_available() else "cpu"
# snapshot_download(repo_id=MODEL_ID, repo_type="model")

print(device)
model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

client = chromadb.PersistentClient(path=DB_PATH)

# 删除可能存在的旧集合，确保我们从一个干净的状态开始
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Deleted old collection: {COLLECTION_NAME}")

collection = client.create_collection(name=COLLECTION_NAME)
print(f"Using device: {device}")
print(f"Database client initialized at: {DB_PATH}")
print(f"Collection '{COLLECTION_NAME}' created.")


# --- 2. 辅助函数：生成图片嵌入 ---

def get_image_embedding(image_path):
    """使用CLIP模型为给定的图片路径生成特征向量"""
    try:
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
            image_features = model.get_image_features(**inputs)
        return image_features.cpu().numpy().flatten().tolist()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


# --- 3. 主程序：遍历并建立索引 ---

def main():
    """主函数，执行建库流程"""
    character_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    # 使用tqdm来创建一个总的进度条
    with tqdm.tqdm(total=sum([len(os.listdir(os.path.join(DATA_DIR, d))) for d in character_folders]),
                   desc="Building Index") as pbar:
        for character_name in character_folders:
            character_path = os.path.join(DATA_DIR, character_name)

            for image_name in os.listdir(character_path):
                # 关键：只索引参考图，跳过目标图'T.png'
                if image_name.upper() == "T.PNG" or image_name.upper() == "D.PNG" or image_name.upper() == "信息.JSON":
                    pbar.update(1)  # 更新进度条，但跳过处理
                    continue

                image_path = os.path.join(character_path, image_name)
                embedding = get_image_embedding(image_path)

                if embedding:
                    # 使用图片路径作为唯一ID，避免重复
                    unique_id = image_path

                    collection.add(
                        ids=[unique_id],
                        embeddings=[embedding],
                        metadatas=[{"character_name": character_name}]
                    )
                pbar.update(1)  # 更新进度条


    print(f"Total items in collection '{COLLECTION_NAME}': {collection.count()}")


if __name__ == '__main__':
    main()
