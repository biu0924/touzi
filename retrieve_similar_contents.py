from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np

model_name = "models/Dmeta-embedding-zh"
index_path = "rag_data/hnsw_index_konwledgebase.index"
csv_path = "rag_data/knowledgebase.csv"

# 1. 加载模型和索引
model = SentenceTransformer(model_name)
index = faiss.read_index(index_path)

# 2. 加载原始文档
df = pd.read_csv(csv_path)

def retrieve_similar_contents(user_input: str,
                              top_k: int = 3,
                              threshold: float = 0.35) -> list:
    # 3. 对用户输入进行向量化
    query_vector = model.encode([user_input])
    # query_vector = np.array([query_vector]).astype('float32')

    # 4. 进行相似度检索
    D, I = index.search(query_vector, k=top_k)

    # 5. 根据阈值筛选结果
    valid_indices = []
    for i, (distance, idx) in enumerate(zip(D[0], I[0])):
        if distance <= threshold:  # 注意：这里使用小于等于，因为距离越小表示越相似
            valid_indices.append(idx)

    # 6. 如果没有有效结果，返回False
    if not valid_indices:
        return False

    # 7. 从原始文档中获取内容
    retrieved_contents = []
    for idx in valid_indices:
        content = f"检索内容[{len(retrieved_contents) + 1}] --> {df.iloc[idx, 0]} | {df.iloc[idx, 1]}"  # 假设文档内容在'content'列
        retrieved_contents.append(content)

    return retrieved_contents

if __name__ == '__main__':
    query = "信用卡逾期怎么办"
    results = retrieve_similar_contents(query)

    if results is False:
        print("没有找到相似度足够高的内容")
    else:
        print("检索到的相似内容：")
        for i, content in enumerate(results, 1):
            print(f"{i}. {content}")
