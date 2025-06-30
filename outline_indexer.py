import os
import requests

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# === НАСТРОЙКИ ===
OUTLINE_API_KEY = os.environ["OUTLINE_API_KEY"]
OUTLINE_API_URL = os.environ["OUTLINE_API_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")


# === ЗАГРУЗКА ДОКУМЕНТОВ ИЗ ВСЕХ КОЛЛЕКЦИЙ OUTLINE ===
def fetch_outline_documents():
    docs = []
    headers = {
        "Authorization": f"Bearer {OUTLINE_API_KEY}",
        "Content-Type": "application/json"
    }

    # Получаем список коллекций
    print("📁 Получаем список коллекций...")
    collections_resp = requests.post(
        f"{OUTLINE_API_URL}/collections.list",
        headers=headers
    )

    collections = collections_resp.json().get("data", [])
    print(f"✅ Найдено коллекций: {len(collections)}")

    for col in collections:
        collection_id = col["id"]
        print(f"🔍 Обрабатываем коллекцию: {col['name']} ({collection_id})")
        page = 1
        while True:
            response = requests.post(
                f"{OUTLINE_API_URL}/documents.list",
                headers=headers,
                json={"collectionId": collection_id, "limit": 100, "offset": (page - 1) * 100}
            )
            if response.status_code != 200:
                print("❌ Ошибка при получении документов:", response.text)
                break

            results = response.json().get("data", [])
            if not results:
                break

            for doc in results:
                if doc.get("archived") or not doc.get("text"):
                    continue
                docs.append(Document(
                    text=doc["text"],
                    metadata={"title": doc["title"], "url": doc.get("url", "")}
                ))

            page += 1

    return docs


# === СОЗДАНИЕ ИЛИ ОБНОВЛЕНИЕ ИНДЕКСА ===
def build_or_update_index(docs):
    persist_dir = "./storage"
    try:
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        index = load_index_from_storage(storage_context)
        print("📦 Существующий индекс загружен. Обновляем...")
        index.insert_documents(docs)
        index.storage_context.persist(persist_dir=persist_dir)
    except Exception as e:
        print("📦 Индекса нет. Создаём новый...")
        index = VectorStoreIndex.from_documents(docs)
        index.storage_context.persist(persist_dir=persist_dir)

    print("✅ Индекс сохранён")


# === ТОЧКА ВХОДА ===
if __name__ == "__main__":
    print("🔄 Загружаем документы из Outline...")
    documents = fetch_outline_documents()
    print(f"📄 Загружено документов: {len(documents)}")

    if documents:
        print("⚙️ Обновляем или создаём индекс...")
        build_or_update_index(documents)
    else:
        print("❌ Документы не найдены — индексация не выполнена.")


