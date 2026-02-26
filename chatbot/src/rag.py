from typing import Any

import asyncio
import logging
import time
from uuid import uuid4

import aiohttp
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.settings import BASE_DIR

BASE_URL = "http://localhost:8000"
CHROMA_PATH = BASE_DIR / ".chroma"
INDEX_NAME = "main-index"

logger = logging.getLogger(__name__)

client = chromadb.PersistentClient(CHROMA_PATH)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=50, length_function=len
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def encode_texts(texts: list[str]) -> list[list[float]]:
    """Преобразование текста в ембединг вектор

    :param texts: Тексты, которые нужно преобразовать в ембединги.
    :returns: Массив ембедингов.
    """

    logger.info("%s send request to encode text ...", f"{BASE_URL}/embeddings")
    timeout = aiohttp.ClientTimeout(total=120 * 5)
    async with aiohttp.ClientSession(base_url=BASE_URL, timeout=timeout) as session, session.post(
        url="/embeddings", json={"texts": texts}, headers={"Content-Type": "application/json"}
    ) as response:
        response.raise_for_status()
        data = await response.json()
        if data.get("embeddings") is None:
            raise ValueError("Missing embeddings values in JSON response!")
        return data["embeddings"]


async def index_document(
        text: str, metadata: dict[str, Any] | None = None, batch_size: int = 10
) -> list[str]:
    """Индексация и добавление документа в семантический индекс.

    :param text: Текст документа.
    :param metadata: Мета-информация документа.
    :param batch_size: Размер батча для векторизации.
    :returns: Идентификаторы чанков в индексе.
    """

    if not text.strip():
        logger.warning("Attempted to index empty text!")
        return []
    start_time = time.monotonic()
    logger.info("Starting index document text, length %s characters", len(text))
    collection = client.get_or_create_collection(INDEX_NAME)
    chunks = splitter.split_text(text)
    ids = [str(uuid4()) for _ in range(len(chunks))]
    # embeddings = await encode_texts(chunks)
    embeddings = []
    total_chunks = len(chunks)
    num_batches = (total_chunks + batch_size - 1) // batch_size
    logger.info("Splitting into %s batches (batch size %s)", num_batches, batch_size)
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = await encode_texts(batch_chunks)
        if len(batch_embeddings) != len(batch_chunks):
            raise ValueError(f"Expected {len(batch_chunks)} embeddings, got {len(embeddings)}!")
        embeddings.extend(batch_embeddings)
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[metadata.copy() for _ in range(len(chunks))],
    )
    elapsed_time = round(time.monotonic() - start_time, 2)
    logger.info(
        "Finished indexing text, time %s seconds", elapsed_time)
    return ids


async def retrieve_documents(
        query: str,
        metadata_filter: dict[str, Any] | None = None,
        search_string: str | None = None,
        n_results: int = 10,
) -> list[str]:
    """Извлечение релевантных документов из семантического индекса.

    :param query: Запрос для поиска.
    :param metadata_filter: Метаданные для фильтрации, пример: `{"source": "my_file.pdf"}`.
    :param search_string: Подстрока для поиска.
    :param n_results: Количество извлекаемых документов.
    """

    collection = client.get_collection(INDEX_NAME)
    logger.info("Retrieving for query: '%s...'", query[:50])
    params = {}
    embeddings = await encode_texts([query])
    params["query_embeddings"] = embeddings
    if metadata_filter is not None:
        params["where"] = metadata_filter
    if search_string is not None:
        params["where_document"] = {"$contains": search_string}
    params["n_results"] = n_results
    result = collection.query(
        **params, include=["documents", "metadatas", "distances"]
    )
    return [
        (
            f"**Relevance score:** {round(distance, 2)}\n"
            f"**Source:** {metadata.get('source', '')}\n"
            f"**Category:** {metadata.get('category', '')}\n"
            f"**Chapter:** {metadata.get('chapter', '')}\n"
            "**Document:**\n"
            f"{document}"
        )
        for document, metadata, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=False
        )
    ]
