import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from datetime import datetime

# ==== Настройки Bitrix ====
BITRIX_WEBHOOK_URL = ""
ENTITY_TYPE_ID = 155  

BITRIX_FIELDS = {
    "title": "title",                         # Название обращения
    "category": "UF_CRM_CATEGORY",            # Категория обращения
    "chat_link": "UF_CRM_CHAT_LINK",          # Ссылка на чат
    "status": "UF_CRM_STATUS",                # Статус (начальное значение)
    "date": "UF_CRM_DATETIME"                 # Дата и время обращения
}

# ==== FastAPI init ====
app = FastAPI()

# ==== Модель запроса ====
class SupportRequest(BaseModel):
    category: str
    chat_id: str  # telegram chat id или ссылка

# ==== Вебхук создания обращения ====
@app.post("/create_support")
async def create_support(request: SupportRequest):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    payload = {
        "entityTypeId": ENTITY_TYPE_ID,
        "fields": {
            BITRIX_FIELDS["title"]: f"Обращение: {request.category}",
            BITRIX_FIELDS["category"]: request.category,
            BITRIX_FIELDS["chat_link"]: request.chat_id,
            BITRIX_FIELDS["status"]: "Открыт",
            BITRIX_FIELDS["date"]: now,
        }
    }

    response = requests.post(BITRIX_WEBHOOK_URL, json=payload)
    if response.ok:
        return {"status": "created"}
    else:
        return {"error": response.text}

# ==== Запуск ====
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
