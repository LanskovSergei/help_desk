import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

from llama_index.core import load_index_from_storage, ServiceContext
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ==== Настройки ====
os.environ["OPENAI_API_KEY"] = "вставь_сюда_твой_OPENAI_ключ"

# ==== Загрузка индекса ====
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

service_context = ServiceContext.from_defaults(
    llm=OpenAI(temperature=0),
    embed_model=OpenAIEmbedding()
)
query_engine = index.as_query_engine(service_context=service_context, similarity_top_k=3)

# ==== FastAPI ====
app = FastAPI()

# ==== Модель запроса ====
class QuestionRequest(BaseModel):
    question: str

# ==== Ответ ====
class AIResponse(BaseModel):
    answer: str
    article_url: Optional[str] = None
    has_answer: bool

# ==== Обработка запроса ====
@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    try:
        response = query_engine.query(request.question)

        # Пытаемся получить ссылку на статью, если есть в metadata
        try:
            source_node = response.source_nodes[0]
            article_url = source_node.node.metadata.get("url")
        except Exception:
            article_url = None

        return AIResponse(
            answer=str(response),
            article_url=article_url,
            has_answer=bool(str(response).strip())
        )
    except Exception as e:
        return AIResponse(
            answer="Произошла ошибка при обработке запроса.",
            article_url=None,
            has_answer=False
        )

# ==== Запуск ====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

