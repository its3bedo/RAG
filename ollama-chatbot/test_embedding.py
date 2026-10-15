
import requests

response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={
        "model": "bge-m3",
        "prompt": "الوثائق المطلوبة لتجديد رخصة المهن"
    },
    timeout=120
)

response.raise_for_status()

embedding = response.json()["embedding"]

print("Embedding length:", len(embedding))
print("First five values:", embedding[:5])
