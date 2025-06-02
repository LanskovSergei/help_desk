from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import os
import uvicorn
from llama_index.core import load_index_from_storage, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import logging

# ==== Настройки ====
os.environ["OPENAI_API_KEY"] = "sk-"
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ==== Логирование ====
logging.basicConfig(filename="log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# ==== Загрузка индекса ====
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=3)

# ==== FastAPI ====
app = FastAPI()

# ==== Модель запроса ====
class QuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class AIResponse(BaseModel):
    answer: str
    article_url: Optional[str] = None
    has_answer: bool

# ==== Ограничение по user_id ====
ALLOWED_USER_IDS = {"123456789"}  # ← сюда вставь ID тестового юзера

@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    logging.info(f"Получен запрос от user_id={request.user_id}: {request.question}")

    if request.user_id not in ALLOWED_USER_IDS:
        logging.warning(f"Запрет для user_id={request.user_id}")
        return AIResponse(
            answer="⛔ Доступ ограничен. Обратитесь к администратору.",
            article_url=None,
            has_answer=False
        )

    try:
        response = query_engine.query(request.question)
        article_url = None
        try:
            source_node = response.source_nodes[0]
            article_url = source_node.node.metadata.get("url")
        except Exception:
            pass

        return AIResponse(
            answer=str(response),
            article_url=article_url,
            has_answer=bool(str(response).strip())
        )
    except Exception as e:
        logging.error(f"Ошибка при обработке запроса: {e}")
        return AIResponse(
            answer="Произошла ошибка при обработке запроса.",
            article_url=None,
            has_answer=False
        )

# ==== Запуск ====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
