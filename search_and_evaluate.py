# search_and_evaluate.py

import os
import torch
import chromadb
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import tqdm

# --- 配置 (必须与建库脚本完全一致) ---
DATA_DIR = "./data"
DB_PATH = "./chroma_db_prod"
COLLECTION_NAME = "character_references"
MODEL_ID = "openai/clip-vit-base-patch32"

# --- 1. 初始化模型和ChromaDB ---

print("Initializing models and database...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_ID)

client = chromadb.PersistentClient(path=DB_PATH)

# 获取已存在的集合
try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Successfully connected to collection '{COLLECTION_NAME}' with {collection.count()} items.")
except Exception as e:
    print(f"Error: Could not get collection '{COLLECTION_NAME}'. Did you run 'build_index.py' first?")
    print(f"Details: {e}")
    exit()


# --- 2. 辅助函数：生成图片嵌入 (与建库脚本一致) ---

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


# --- 3. 主程序：遍历、搜索并评估 ---

def main():
    """主函数，执行搜索和评估流程"""
    print("\nStarting evaluation...")

    correct_predictions = 0
    total_queries = 0

    character_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

    with tqdm.tqdm(total=len(character_folders), desc="Evaluating") as pbar:
        for ground_truth_character in character_folders:
            query_image_path = os.path.join(DATA_DIR, ground_truth_character, "T.png")

            if not os.path.exists(query_image_path):
                pbar.update(1)
                continue

            total_queries += 1
            query_embedding = get_image_embedding(query_image_path)

            if not query_embedding:
                print(f"Skipping evaluation for {query_image_path} due to processing error.")
                pbar.update(1)
                continue

            # 在ChromaDB中查询最相似的1个结果
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )

            if results and results['ids'][0]:
                predicted_character = results['metadatas'][0][0]['character_name']

                if predicted_character == ground_truth_character:
                    correct_predictions += 1
                    # print(f"✅ Correct: '{ground_truth_character}' -> Predicted: '{predicted_character}'")
                else:
                    print(
                        f"❌ Incorrect: Ground Truth was '{ground_truth_character}', but predicted '{predicted_character}'")
            else:
                print(f"⚠️ No result found for query: {query_image_path}")

            pbar.update(1)

    # --- 4. 打印最终的评估报告 ---
    print("\n--- Evaluation Report ---")
    if total_queries > 0:
        accuracy = (correct_predictions / total_queries) * 100
        print(f"Total Queries:      {total_queries}")
        print(f"Correct Predictions:  {correct_predictions}")
        print(f"Accuracy:             {accuracy:.2f}%")
    else:
        print("No target images ('T.png') found to evaluate.")
    print("-------------------------")


if __name__ == '__main__':
    main()
