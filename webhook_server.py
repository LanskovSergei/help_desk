import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import ServiceContext
import uvicorn

# ==== Настройки ====
os.environ["OPENAI_API_KEY"] = ""

# ==== FastAPI init ====
app = FastAPI()

# ==== Загрузка индекса один раз при старте ====
print("🧠 Загружаем индекс...")
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
