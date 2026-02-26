from typing import Literal

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, status
from pydantic import BaseModel

from scripts import init_kb
from src.agent import UserContext, call_agent


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_kb.main()
    yield


app = FastAPI(
    title="AI консультант приёмной комиссии",
    description="""\
    Нейро-консультант приёмной комиссии Тюменского Индустриального Университета.
    Подстраивает свои ответы по ситуацию и описание пользователя.
    """,
    version="0.1.0",
    lifespan=lifespan
)


class UserPrompt(BaseModel):
    """Запрос для генерации ответа"""

    user_id: str
    role: Literal["Школьник", "Абитуриент", "Родитель"] = "Абитуриент"
    purpose: str | None = None
    text: str


@app.post(
    path="/agent",
    status_code=status.HTTP_200_OK,
    summary="Сгенерировать ответ"
)
async def generate_response(prompt: UserPrompt) -> dict[str, str]:
    content = await call_agent(
        prompt.text, context=UserContext(
            user_id=prompt.user_id, role=prompt.role, purpose=prompt.purpose
        )
    )
    return {"text": content}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104
