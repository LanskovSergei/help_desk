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
os.environ["OPENAI_API_KEY"] = "sk-..."

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ==== Загрузка индекса ====
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ==== Движок запросов ====
query_engine = index.as_query_engine(similarity_top_k=3)

# ==== FastAPI ====
app = FastAPI()

# ==== Модели запроса и ответа ====
class QuestionRequest(BaseModel):
    question: str
    user_id: Optional[str] = None

class AIResponse(BaseModel):
    answer: str
    article_url: Optional[str] = None
    has_answer: bool

# ==== Обработка запроса ====
@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    # Логируем запрос
    log_line = f"{datetime.now().isoformat()} | USER_ID: {request.user_id} | Q: {request.question}\n"
    with open("requests.log", "a") as f:
        f.write(log_line)

    try:
        response = query_engine.query(request.question)
        response_text = str(response)

        try:
            source_node = response.source_nodes[0]
            article_url = source_node.node.metadata.get("url")
        except Exception:
            article_url = None

        # Анализируем, есть ли смысловой ответ
        has_answer = not (
            "не дал результатов" in response_text.lower() or
            "информация не найдена" in response_text.lower() or
            len(response_text.strip()) == 0
        )

        return AIResponse(
            answer=response_text,
            article_url=article_url,
            has_answer=has_answer
        )
    except Exception as e:
        return AIResponse(
            answer="Информация не найдена. Хотите поговорить с оператором?",
            article_url=None,
            has_answer=False
        )

# ==== Запуск ====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


