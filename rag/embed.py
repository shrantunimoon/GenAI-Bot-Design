from sentence_transformers import SentenceTransformer
import sqlite3
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

conn = sqlite3.connect("rag/db.sqlite")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    text TEXT,
    embedding BLOB
)
""")

def chunk_text(text, size=200):
    return [text[i:i+size] for i in range(0, len(text), size)]

def embed_and_store():
    for file in os.listdir("data"):
        with open(f"data/{file}", "r") as f:
            text = f.read()

        chunks = chunk_text(text)

        for chunk in chunks:
            emb = model.encode(chunk).tobytes()
            c.execute("INSERT INTO docs (text, embedding) VALUES (?, ?)", (chunk, emb))

    conn.commit()

if __name__ == "__main__":
    embed_and_store()
