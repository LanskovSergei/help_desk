import os
import requests
from llama_index import VectorStoreIndex, Document
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import ServiceContext

# === НАСТРОЙКИ ===
OUTLINE_API_KEY = ""
OUTLINE_API_URL = ""
OPENAI_API_KEY = ""

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# === ЗАГРУЗКА ДОКУМЕНТОВ ИЗ OUTLINE ===
def fetch_outline_documents():
    docs = []
    page = 1
    while True:
        response = requests.post(
            OUTLINE_API_URL,
            headers={"Authorization": f"Bearer {OUTLINE_API_KEY}"},
            json={"limit": 100, "offset": (page - 1) * 100}
        )
        if response.status_code != 200:
            print("Ошибка при получении документов:", response.text)
            break

        data = response.json()
        results = data.get("data", [])
        if not results:
            break

        for doc in results:
            if doc.get("archived") or not doc.get("text"):
                continue
            docs.append(Document(
                text=doc["text"],
                metadata={"title": doc["title"]}
            ))

        page += 1
    return docs

# === СОЗДАНИЕ ИНДЕКСА ===
def build_index(docs):
    service_context = ServiceContext.from_defaults(
        llm=OpenAI(temperature=0),
        embed_model=OpenAIEmbedding()
    )
    index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    index.storage_context.persist(persist_dir="./storage")
    return index

# === ОСНОВНАЯ ТОЧКА ВХОДА ===
if __name__ == "__main__":
    print("🔄 Загружаем документы из Outline...")
    documents = fetch_outline_documents()
    print(f"✅ Загружено документов: {len(documents)}")

    print("⚙️ Строим индекс...")
    build_index(documents)
    print("✅ Индекс сохранён в ./storage")
