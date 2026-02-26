from typing import Literal

import logging
import os

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from .prompts import SUMMARY_PROMPT, SYSTEM_PROMPT
from .rag import retrieve_documents
from .settings import SQLITE_PATH, settings

logger = logging.getLogger(__name__)

model = ChatOpenAI(
    api_key=settings.yandex_cloud_api_key,
    model=settings.qwen3_235b,
    base_url=settings.llm_api_base_url,
    temperature=0.3,
)


class UserContext(BaseModel):
    """Контекстная информация о пользователе"""

    user_id: str
    role: Literal["Школьник", "Абитуриент", "Родитель"] = "Абитуриент"
    purpose: str | None = Field(
        default=None,
        description="Цель пользователя",
        examples=[
            "Поступить на бюджет на направление Прикладная информатика",
            "Найти подходящее для себя направление",
            "Пройти подготовку к ЕГЭ"
        ]
    )


class SearchInput(BaseModel):
    """Входные параметры для писка информации"""

    query: str = Field(description="Запрос для поиска")


@tool(
    "knowledge_search",
    description="Выполняет поиск во внутренней базе знаний",
    args_schema=SearchInput
)
async def knowledge_search(query: str) -> str:
    docs = await retrieve_documents(query)
    return "\n\n".join(docs)


summarization_middleware = SummarizationMiddleware(
    model=model,
    summary_prompt=SUMMARY_PROMPT,
    trigger=("tokens", 8000),
    keep=("messages", 25),
)


async def call_agent(user_prompt: str, context: UserContext) -> str:
    """Вызов AI ассистента приёмной комиссии.

    :param user_prompt: Запрос пользователя.
    :param context: Контекстная информация о пользователе.
    :returns: Текстовый ответ от ассистента.
    """

    async with AsyncSqliteSaver.from_conn_string(os.fspath(SQLITE_PATH)) as checkpointer:
        await checkpointer.setup()
        agent = create_agent(
            model=model,
            system_prompt=SYSTEM_PROMPT.format(role=context.role, purpose=context.purpose),
            middleware=[summarization_middleware],
            tools=[knowledge_search],
            checkpointer=checkpointer
        )
        result = await agent.ainvoke(
            {"messages": [("human", user_prompt)]},
            config={"configurable": {"thread_id": context.user_id}}
        )
    return result["messages"][-1].content
