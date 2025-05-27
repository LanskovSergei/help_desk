import os
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import ServiceContext

# ==== Настройки ====
os.environ["OPENAI_API_KEY"] = ""

# ==== Загружаем индекс из ./storage ====
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ==== Создаём движок запросов ====
service_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0),
    embed_model=OpenAIEmbedding()
)
query_engine = index.as_query_engine(service_context=service_context)

# ==== Цикл вопросов ====
print("🧠 Введите вопрос (или 'exit' для выхода):")
while True:
    user_input = input("❓ Вопрос: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    try:
        response = query_engine.query(user_input)
        print("\n💬 Ответ:\n", response, "\n")
    except Exception as e:
        print("⚠️ Ошибка при обработке запроса:", e)
