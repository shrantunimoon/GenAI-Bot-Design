import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
conn = sqlite3.connect("rag/db.sqlite")

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, top_k=3):
    q_emb = model.encode(query)

    cursor = conn.cursor()
    cursor.execute("SELECT text, embedding FROM docs")

    scores = []

    for text, emb in cursor.fetchall():
        emb = np.frombuffer(emb, dtype=np.float32)
        score = cosine_sim(q_emb, emb)
        scores.append((text, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    return [s[0] for s in scores[:top_k]]
