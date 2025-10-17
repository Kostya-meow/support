from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import delete, select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import models


async def get_conversation_by_chat_id(
    session: AsyncSession,
    telegram_chat_id: int,
) -> Optional[models.Conversation]:
    result = await session.execute(
        select(models.Conversation).where(
            models.Conversation.telegram_chat_id == telegram_chat_id
        )
    )
    return result.scalar_one_or_none()


async def create_conversation(
    session: AsyncSession,
    telegram_chat_id: int,
    title: Optional[str],
) -> models.Conversation:
    conversation = models.Conversation(telegram_chat_id=telegram_chat_id, title=title)
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def list_conversations(
    session: AsyncSession,
    *,
    operator_requested_only: bool = True,
    archived: bool = False,
) -> list[models.Conversation]:
    stmt = select(models.Conversation).order_by(models.Conversation.updated_at.desc())
    if operator_requested_only:
        stmt = stmt.where(models.Conversation.operator_requested.is_(True))

    stmt = stmt.where(models.Conversation.is_archived.is_(archived))
    result = await session.execute(stmt)
    return result.scalars().all()


async def list_messages_for_conversation(
    session: AsyncSession,
    conversation_id: int,
) -> list[models.Message]:
    result = await session.execute(
        select(models.Message)
        .where(models.Message.conversation_id == conversation_id)
        .order_by(models.Message.created_at)
    )
    return result.scalars().all()


async def add_message(
    session: AsyncSession,
    conversation_id: int,
    sender: str,
    text: str,
    telegram_message_id: Optional[int] = None,
    is_system: bool = False,
    created_at: Optional[datetime] = None,
) -> models.Message:
    message = models.Message(
        conversation_id=conversation_id,
        sender=sender,
        text=text,
        telegram_message_id=telegram_message_id,
        is_system=is_system,
    )
    # Не передавать created_at, всегда использовать серверное время
    session.add(message)
    conversation = await session.get(models.Conversation, conversation_id)
    if conversation:
        conversation.updated_at = datetime.utcnow()
        if sender != "operator":
            conversation.unread_count = (conversation.unread_count or 0) + 1
    await session.commit()
    await session.refresh(message)
    return message


async def set_operator_requested(
    session: AsyncSession,
    conversation_id: int,
    requested: bool,
) -> Optional[models.Conversation]:
    conversation = await session.get(models.Conversation, conversation_id)
    if conversation is None:
        return None
    if conversation.operator_requested != requested:
        conversation.operator_requested = requested
        conversation.updated_at = datetime.utcnow()
        if not requested:
            conversation.unread_count = 0
            conversation.is_archived = (
                True  # Автоматически архивируем завершенные диалоги
            )
        await session.commit()
        await session.refresh(conversation)
    return conversation


async def set_archived_status(
    session: AsyncSession,
    conversation_id: int,
    archived: bool,
) -> Optional[models.Conversation]:
    conversation = await session.get(models.Conversation, conversation_id)
    if conversation is None:
        return None
    if conversation.is_archived != archived:
        conversation.is_archived = archived
        conversation.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(conversation)
    return conversation


async def reset_unread(
    session: AsyncSession, conversation_id: int
) -> Optional[models.Conversation]:
    conversation = await session.get(models.Conversation, conversation_id)
    if conversation is None:
        return None
    if conversation.unread_count:
        conversation.unread_count = 0
        await session.commit()
        await session.refresh(conversation)
    return conversation


async def replace_knowledge_entries(
    session: AsyncSession,
    entries: Iterable[tuple[str, str, bytes]],
) -> None:
    await session.execute(delete(models.KnowledgeEntry))
    session.add_all(
        [
            models.KnowledgeEntry(question=question, answer=answer, embedding=embedding)
            for question, answer, embedding in entries
        ]
    )
    await session.commit()


async def load_knowledge_entries(session: AsyncSession) -> list[models.KnowledgeEntry]:
    result = await session.execute(
        select(models.KnowledgeEntry).order_by(models.KnowledgeEntry.id)
    )
    return result.scalars().all()


async def count_knowledge_entries(session: AsyncSession) -> int:
    result = await session.execute(
        select(func.count()).select_from(models.KnowledgeEntry)
    )
    count = result.scalar_one()
    return int(count or 0)


async def get_random_knowledge_entry(
    session: AsyncSession,
) -> models.KnowledgeEntry | None:
    """Получить случайную запись из базы знаний"""
    result = await session.execute(
        select(models.KnowledgeEntry).order_by(func.random()).limit(1)
    )
    return result.scalar_one_or_none()
