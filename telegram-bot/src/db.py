from typing import Literal

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, func, select
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .settings import SQLALCHEMY_URL

engine = create_async_engine(url=SQLALCHEMY_URL, echo=True)
sessionmaker = async_sessionmaker(
    engine, class_=AsyncSession, autoflush=False, expire_on_commit=False
)


class Base(AsyncAttrs, DeclarativeBase):
    __abstract__ = True

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


async def create_tables() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


class User(Base):
    __tablename__ = "users"

    username: Mapped[str | None] = mapped_column(unique=True, nullable=True)
    role: Mapped[str]
    purpose: Mapped[str | None] = mapped_column(nullable=True)


async def create_user(
        user_id: int,
        username: str,
        role: Literal["Школьник", "Абитуриент", "Родитель"] = "Абитуриент",
        purpose: str | None = None
) -> None:
    async with sessionmaker() as session:
        user = User(id=user_id, username=username, role=role, purpose=purpose)
        session.add(user)
        await session.commit()


async def get_user(user_id: int) -> User | None:
    async with sessionmaker() as session:
        stmt = select(User).where(User.id == user_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
