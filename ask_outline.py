import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
from datetime import datetime

from llama_index.core import load_index_from_storage, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ==== Config ====
os.environ["OPENAI_API_KEY"] = "sk-proj-..." 

# Разрешённые пользователи (для теста)
ALLOWED_USER_IDS = {"123456789"}  

# Настройки модели и эмбеддинга
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ==== Загрузка индекса ====
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=3)

# ==== FastAPI App ====
app = FastAPI()

# ==== Модели ====
class QuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class AIResponse(BaseModel):
    answer: str
    article_url: Optional[str] = None
    has_answer: bool
    user_id: Optional[str] = None  # добавили это поле

# ==== Основной эндпоинт ====
@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    # Логируем
    log_line = f"{datetime.now().isoformat()} | USER_ID: {request.user_id} | Q: {request.question}\n"
    with open("requests.log", "a") as f:
        f.write(log_line)

    # Проверка доступа
    if request.user_id not in ALLOWED_USER_IDS:
        return AIResponse(
            answer="Access denied. You are not authorized to use this service.",
            article_url=None,
            has_answer=False,
            user_id=request.user_id
        )

    try:
        response = query_engine.query(request.question)
        response_text = str(response).strip().lower()

        failure_phrases = [
            "no information", "no relevant", "no results",
            "не найдено", "не удалось найти", "нет информации", "в предоставленном контексте не"
        ]

        if any(phrase in response_text for phrase in failure_phrases):
            return AIResponse(
                answer="Information not found. Would you like to talk to an operator?",
                article_url=None,
                has_answer=False,
                user_id=request.user_id
            )

        try:
            article_url = response.source_nodes[0].node.metadata.get("url")
        except Exception:
            article_url = None

        return AIResponse(
            answer=str(response),
            article_url=article_url,
            has_answer=True,
            user_id=request.user_id  # ← обязательно возвращаем
        )

    except Exception:
        return AIResponse(
            answer="Information not found. Would you like to talk to an operator?",
            article_url=None,
            has_answer=False,
            user_id=request.user_id
        )

# ==== Запуск сервера ====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)




