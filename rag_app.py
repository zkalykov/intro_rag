import os
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import OpenAI
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables (API key)
load_dotenv()
API_KEY = os.getenv("META_LLAMA_API_KEY")
BASE_URL = "https://api.llama.com/compat/v1/"

if not API_KEY:
    raise ValueError("❌ API key not found in .env file (META_LLAMA_API_KEY).")

# Initialize LLM client
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Load city data
with open("capital_cities.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [f"{c['city']}, {c['country']}: {c['description']}" for c in data]
ids = [f"id_{i}" for i in range(len(texts))]

# Load embedding model (shared by both Chroma and custom use)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Set up ChromaDB client and collection (in-memory)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="cities", embedding_function=embedding_fn)

# Add documents to ChromaDB
collection.add(
    documents=texts,
    ids=ids,
    metadatas=[{"city": data[i]['city'], "country": data[i]['country']} for i in range(len(data))]
)

# User input loop
print("🌍 Ask about a capital city (type 'q' to quit):")
while True:
    query = input("🧠 Question: ")
    if query.strip().lower() == "q":
        print("👋 Exiting.")
        break

    # Query ChromaDB (top 2 results)
    results = collection.query(
        query_texts=[query],
        n_results=2
    )

    if not results["documents"] or not results["documents"][0]:
        print("🤖 No relevant information found.")
        continue

    # Prepare context for GPT
    context = "\n".join(results["documents"][0])

    try:
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who only answers using the given context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
            ],
            temperature=0.7,
            max_tokens=500
        )

        print("🤖 Answer:", response.choices[0].message.content.strip())

    except Exception as e:
        print("❌ Error from API:", e)
