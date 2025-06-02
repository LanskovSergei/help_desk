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

# ==== Настройки ====
os.environ["OPENAI_API_KEY"] = "sk-proj-..." 

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ==== Загрузка индекса ====
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ==== Движок запросов ====
query_engine = index.as_query_engine(similarity_top_k=3)

# ==== FastAPI ====
app = FastAPI()

# ==== Модели ====
class QuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class AIResponse(BaseModel):
    answer: str
    article_url: Optional[str] = None
    has_answer: bool

# ==== Эндпоинт ====
@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    log_line = f"{datetime.now().isoformat()} | USER_ID: {request.user_id} | Q: {request.question}\n"
    with open("requests.log", "a") as f:
        f.write(log_line)

    try:
        response = query_engine.query(request.question)
        answer_text = str(response).strip()

        # Проверка на пустой или сгенерированный ответ без ссылки
        try:
            source_node = response.source_nodes[0]
            article_url = source_node.node.metadata.get("url")
        except Exception:
            article_url = None

        if not answer_text or not article_url:
            return AIResponse(
                answer="Информация не найдена. Хотите поговорить с оператором?",
                article_url=None,
                has_answer=False
            )

        return AIResponse(
            answer=answer_text,
            article_url=article_url,
            has_answer=True
        )
    except Exception as e:
        return AIResponse(
            answer="Информация не найдена. Хотите поговорить с оператором?",
            article_url=None,
            has_answer=False
        )

# ==== Запуск сервера ====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

