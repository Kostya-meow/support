from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from sqlalchemy import delete, select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app import models


# ========== TICKETS ==========

async def get_open_ticket_by_chat_id(
    session: AsyncSession,
    telegram_chat_id: int,
) -> Optional[models.Ticket]:
    """Получить открытую заявку для данного чата."""
    result = await session.execute(
        select(models.Ticket).where(
            and_(
                models.Ticket.telegram_chat_id == telegram_chat_id,
                models.Ticket.status.in_([models.TicketStatus.OPEN, models.TicketStatus.IN_PROGRESS])
            )
        )
    )
    return result.scalar_one_or_none()


async def create_ticket(
    session: AsyncSession,
    telegram_chat_id: int,
    title: Optional[str] = None,
) -> models.Ticket:
    """Создать новую заявку."""
    ticket = models.Ticket(
        telegram_chat_id=telegram_chat_id,
        title="Временная заявка",  # Временный заголовок
        status=models.TicketStatus.OPEN
    )
    session.add(ticket)
    await session.commit()
    await session.refresh(ticket)
    
    # Обновляем заголовок с номером заявки
    if title:
        ticket.title = f"Заявка #{ticket.id} - {title}"
    else:
        ticket.title = f"Заявка #{ticket.id} - Пользователь {telegram_chat_id}"
    await session.commit()
    
    return ticket


async def list_tickets(
    session: AsyncSession,
    *,
    status: Optional[models.TicketStatus] = None,
    archived: bool = False,
) -> list[models.Ticket]:
    """Получить список заявок."""
    stmt = select(models.Ticket).order_by(models.Ticket.updated_at.desc())
    
    if status:
        stmt = stmt.where(models.Ticket.status == status)
    elif archived:
        # Архив - закрытые и архивированные заявки
        stmt = stmt.where(models.Ticket.status.in_([models.TicketStatus.CLOSED, models.TicketStatus.ARCHIVED]))
    else:
        # Активные заявки - только открытые и в работе
        stmt = stmt.where(models.Ticket.status.in_([models.TicketStatus.OPEN, models.TicketStatus.IN_PROGRESS]))
    
    result = await session.execute(stmt)
    return result.scalars().all()


async def get_ticket_by_id(
    session: AsyncSession,
    ticket_id: int,
) -> Optional[models.Ticket]:
    """Получить заявку по ID."""
    return await session.get(models.Ticket, ticket_id)


async def update_ticket_status(
    session: AsyncSession,
    ticket_id: int,
    status: models.TicketStatus,
) -> Optional[models.Ticket]:
    """Обновить статус заявки."""
    ticket = await session.get(models.Ticket, ticket_id)
    if ticket is None:
        return None
    
    ticket.status = status
    ticket.updated_at = datetime.utcnow()
    
    if status == models.TicketStatus.CLOSED:
        ticket.closed_at = datetime.utcnow()
    elif status == models.TicketStatus.ARCHIVED:
        if not ticket.closed_at:
            ticket.closed_at = datetime.utcnow()
    
    await session.commit()
    await session.refresh(ticket)
    return ticket


# ========== MESSAGES ==========

async def add_message(
    session: AsyncSession,
    ticket_id: int,
    sender: str,
    text: str,
    telegram_message_id: Optional[int] = None,
    is_system: bool = False,
    created_at: Optional[datetime] = None,
) -> models.Message:
    """Добавить сообщение в заявку."""
    message = models.Message(
        ticket_id=ticket_id,
        sender=sender,
        text=text,
        telegram_message_id=telegram_message_id,
        is_system=is_system,
    )
    if created_at:
        message.created_at = created_at
    session.add(message)
    
    # Обновляем время последнего обновления заявки
    ticket = await session.get(models.Ticket, ticket_id)
    if ticket:
        ticket.updated_at = datetime.utcnow()
    
    await session.commit()
    await session.refresh(message)
    return message


async def list_messages_for_ticket(
    session: AsyncSession,
    ticket_id: int,
    include_system: bool = True,
) -> list[models.Message]:
    """Получить все сообщения заявки."""
    stmt = select(models.Message).where(models.Message.ticket_id == ticket_id)
    
    if not include_system:
        stmt = stmt.where(models.Message.is_system == False)
    
    stmt = stmt.order_by(models.Message.created_at)
    result = await session.execute(stmt)
    return result.scalars().all()


# ========== KNOWLEDGE ==========

async def replace_knowledge_entries(
    session: AsyncSession,
    entries: Iterable[tuple[str, str, bytes]],
) -> None:
    """Заменить все записи в базе знаний."""
    await session.execute(delete(models.KnowledgeEntry))
    session.add_all(
        [
            models.KnowledgeEntry(question=question, answer=answer, embedding=embedding)
            for question, answer, embedding in entries
        ]
    )
    await session.commit()


async def load_knowledge_entries(session: AsyncSession) -> list[models.KnowledgeEntry]:
    """Загрузить все записи базы знаний."""
    result = await session.execute(select(models.KnowledgeEntry).order_by(models.KnowledgeEntry.id))
    return result.scalars().all()


async def count_knowledge_entries(session: AsyncSession) -> int:
    """Подсчитать количество записей в базе знаний."""
    result = await session.execute(select(func.count()).select_from(models.KnowledgeEntry))
    count = result.scalar_one()
    return int(count or 0)