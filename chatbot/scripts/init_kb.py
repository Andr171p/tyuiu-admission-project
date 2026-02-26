# Скрипт для инициализации и наполнения базы знаний

import asyncio
import logging
import time

import anyio
from markitdown import MarkItDown
from tqdm import tqdm

from src.rag import CHROMA_PATH, index_document
from src.settings import BASE_DIR

DATA_DIR = BASE_DIR / "data"

logger = logging.getLogger(__name__)


def convert_document_to_md(file_path) -> str:
    """Конвертирует документ (.pptx, .pdf, .docx, .xlsx) в Markdown текст"""

    md = MarkItDown()
    result = md.convert(file_path)
    return result.text_content


async def main() -> None:
    if await anyio.Path(CHROMA_PATH).exists():
        return
    files = [
        file_path async for file_path in anyio.Path(DATA_DIR).rglob("*")
        if await file_path.is_file()
    ]
    logger.info("Found %s files to indexing", len(files))
    for file_path in tqdm(files, desc="Indexing files", unit="file", total=len(files)):
        start_time = time.monotonic()
        logger.info("Start processing `%s` file ...", file_path)
        if file_path.suffix != "md":
            logger.info("Converting `%s` file to Markdown text ...", file_path.suffix)
            md_text = convert_document_to_md(str(file_path))
        else:
            md_text = await file_path.read_text(encoding="utf-8")
        resolved_path = await file_path.resolve()
        await index_document(
            md_text,
            metadata={
                "category": "Приёмная компания 2026",
                "source": file_path.name,
                "chapter": resolved_path.parent.name
            }
        )
        elapsed_time = time.monotonic() - start_time
        logger.info(
            "Successfully indexing `%s` file, elapsed %s seconds",
            file_path, round(elapsed_time, 2)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
