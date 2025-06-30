import os
import requests
from llama_index.core import VectorStoreIndex, Document, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# ======= Настройки =======
OUTLINE_API_URL = "https://outline.taliaslimbot.com/api"
OUTLINE_API_KEY = "ol_"
OPENAI_API_KEY = "sk-"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
HEADERS = {"Authorization": f"Bearer {OUTLINE_API_KEY}"}

# ======= Получение всех документов из всех коллекций =======
def fetch_all_documents():
    print("📁 Получаем список коллекций...")
    collections = requests.post(
        f"{OUTLINE_API_URL}/collections.list",
        headers=HEADERS
    ).json().get("data", [])
    print(f"✅ Найдено коллекций: {len(collections)}")

    all_docs = []

    for col in collections:
        col_id = col["id"]
        col_name = col["name"]
        print(f"🔍 Обрабатываем коллекцию: {col_name} ({col_id})")

        page = 0
        while True:
            resp = requests.post(
                f"{OUTLINE_API_URL}/documents.list",
                headers=HEADERS,
                json={"collectionId": col_id, "limit": 100, "offset": page * 100}
            )
            results = resp.json().get("data", [])
            if not results:
                break

            for doc in results:
                doc_id = doc["id"]
                title = doc["title"]

                export_resp = requests.post(
                    f"{OUTLINE_API_URL}/documents.export",
                    headers=HEADERS,
                    json={"id": doc_id}
                )
                if export_resp.status_code != 200:
                    print(f"❌ Ошибка при экспорте документа: {title}")
                    continue

                text = export_resp.json().get("data", "").strip()
                if not text:
                    print(f"⚠️ Документ пустой: {title}")
                    continue

                all_docs.append(Document(
                    text=text,
                    metadata={"title": title, "collection": col_name}
                ))

            page += 1

    print(f"📄 Загружено документов: {len(all_docs)}")
    return all_docs

# ======= Построение индекса =======
def build_index(docs):
    Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir="./storage")
    return index

# ======= Точка входа =======
if __name__ == "__main__":
    print("🔄 Загружаем документы из Outline...")
    documents = fetch_all_documents()

    if documents:
        print("⚙️ Строим индекс...")
        build_index(documents)
        print("✅ Индекс сохранён в ./storage")
    else:
        print("❌ Нет документов для индексации.")



