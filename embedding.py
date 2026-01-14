import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import psycopg2
from sentence_transformers import SentenceTransformer

# 1. 初始化 embedding 模型（384维）
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 2. 字段列表
fields = ["姓名", "手机号", "邮箱", "部门"]

# 3. 生成 embeddings
embeddings = model.encode(fields)  # shape: (4, 384)

# 4. 连接 PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="123456"
)
cur = conn.cursor()

# 5. 插入数据
for field, emb in zip(fields, embeddings):
    # 将 numpy array 转为 pgvector 接受的字符串格式：'[1,2,3]'
    emb_str = "[" + ",".join(map(str, emb.tolist())) + "]"
    cur.execute(
        "INSERT INTO form_fields (field_name, embedding) VALUES (%s, %s) "
        "ON CONFLICT (field_name) DO UPDATE SET embedding = EXCLUDED.embedding;",
        (field, emb_str)
    )

# 6. 提交并关闭
conn.commit()
cur.close()
conn.close()

print("✅ Embeddings inserted successfully!")