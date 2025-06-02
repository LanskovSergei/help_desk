import os
import json
from datetime import datetime
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional
import uvicorn

from llama_index.core import load_index_from_storage, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# === Настройки ===
os.environ["OPENAI_API_KEY"] = "sk-..."
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# === Индекс ===
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=3)

# === App ===
app = FastAPI()

# === Логгер ===
def log_request(data):
    with open("bitrix_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} — {json.dumps(data, ensure_ascii=False)}\n")

# === Bitrix webhook ===
TEST_USER_ID = 123456  # ID тестового юзера

@app.post("/bitrix_webhook")
async def receive_bitrix(request: Request):
    data = await request.json()
    log_request(data)  # логируем всё

    user_id = data.get("user", {}).get("id")
    message = data.get("message", "")

    # Пропускаем всех кроме тестового пользователя
    if user_id != TEST_USER_ID:
        return {"status": "ignored", "reason": "not a test user"}

    try:
        response = query_engine.query(message)
        return {
            "status": "ok",
            "answer": str(response)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# === Для ручного запроса ===
class QuestionRequest(BaseModel):
    question: str

class AIResponse(BaseModel):
    answer: str
    article_url: Optional[str] = None
    has_answer: bool

@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    try:
        response = query_engine.query(request.question)

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
    except Exception:
        return AIResponse(
            answer="Произошла ошибка при обработке запроса.",
            article_url=None,
            has_answer=False
        )

# === Запуск ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
