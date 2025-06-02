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

# ==== API Key and LLM Settings ====
os.environ["OPENAI_API_KEY"] = "sk-proj-..."

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# ==== Load Knowledge Base Index ====
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# ==== Initialize Query Engine ====
query_engine = index.as_query_engine(similarity_top_k=3)

# ==== FastAPI App ====
app = FastAPI()

# ==== Allowed test user(s) ====
ALLOWED_USER_IDS = {"123456789"}  # Add more user_ids as needed

# ==== Request and Response Models ====
class QuestionRequest(BaseModel):
    question: str
    user_id: str

class AIResponse(BaseModel):
    answer: str
    article_url: Optional[str] = None
    has_answer: bool

# ==== API Endpoint ====
@app.post("/ask", response_model=AIResponse)
async def ask_ai(request: QuestionRequest):
    # Log the incoming request
    log_line = f"{datetime.now().isoformat()} | USER_ID: {request.user_id} | Q: {request.question}\n"
    with open("requests.log", "a") as f:
        f.write(log_line)

    # Check if user is allowed to test
    if request.user_id not in ALLOWED_USER_IDS:
        return AIResponse(
            answer="Access denied. You are not authorized to use this service.",
            article_url=None,
            has_answer=False
        )

    try:
        response = query_engine.query(request.question)
        response_text = str(response).strip()

        # If no meaningful response
        if not response_text or "no results" in response_text.lower() or "no information" in response_text.lower():
            return AIResponse(
                answer="Information not found. Would you like to talk to an operator?",
                article_url=None,
                has_answer=False
            )

        try:
            article_url = response.source_nodes[0].node.metadata.get("url")
        except Exception:
            article_url = None

        return AIResponse(
            answer=response_text,
            article_url=article_url,
            has_answer=True
        )

    except Exception:
        return AIResponse(
            answer="Information not found. Would you like to talk to an operator?",
            article_url=None,
            has_answer=False
        )

# ==== Run the server ====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


