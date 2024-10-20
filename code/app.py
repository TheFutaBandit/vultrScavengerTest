import os
import requests

VULTR_API_KEY = os.getenv("VULTR_API_KEY")
COLLECTION_ID = "myvultrcollect"

def create_collection_item(content, description=""):
    url = f"https://api.vultrinference.com/v1/vector_store/{COLLECTION_ID}/items"
    headers = {
        "Authorization": f"Bearer {VULTR_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "content": content,
        "description": description
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def rag_chat_completion(user_input):
    url = "https://api.vultrinference.com/v1/chat/completions/RAG"
    headers = {
        "Authorization": f"Bearer {VULTR_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "collection": COLLECTION_ID,
        "model": "llama2-7b-chat-Q5_K_M",
        "messages": [
            {"role": "user", "content": user_input}
        ],
        "max_tokens": 512,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

if __name__ == "__main__":
    user_input = input("Ask a question: ")
    response = rag_chat_completion(user_input)
    print("Response:", response)
