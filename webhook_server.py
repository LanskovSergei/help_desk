import os
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from llama_index.core import load_index_from_storage, ServiceContext
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ==== Настройки ====
os.environ["OPENAI_API_KEY"] = "вставь_сюда_твой_openai_api_key"

# ==== FastAPI init ====
app = FastAPI()

# ==== Загружаем индекс ====
print("🧠 Загружаем индекс из ./storage...")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

service_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0),
    embed_model=OpenAIEmbedding()
)
query_engine = index.as_query_engine(service_context=service_context)

# ==== Модель запроса ====
class AskRequest(BaseModel):
    question: str

# ==== Вебхук ====
@app.post("/ask")
async def ask(request: AskRequest):
    try:
        response = query_engine.query(request.question)
        return {"answer": str(response)}
    except Exception as e:
        return {"error": str(e)}

# ==== Запуск ====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

