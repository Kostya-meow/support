from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import BigInteger, Boolean, Column, DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class TicketStatus(str, Enum):
    OPEN = "open"           # Открыта
    IN_PROGRESS = "in_progress"  # В работе
    CLOSED = "closed"       # Закрыта
    ARCHIVED = "archived"   # Архивирована


class Ticket(Base):
    """Заявка в службу поддержки"""
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    telegram_chat_id = Column(BigInteger, index=True, nullable=False)
    title = Column(String(255), nullable=True)
    status = Column(String(20), default=TicketStatus.OPEN, nullable=False)
    priority = Column(String(10), default="medium", nullable=False)  # low, medium, high
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )
    closed_at = Column(DateTime, nullable=True)

    # Связи
    messages = relationship(
        "Message",
        back_populates="ticket",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )


class Message(Base):
    """Сообщение в заявке"""
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False, index=True)
    sender = Column(String(32), nullable=False)  # user, bot, operator
    text = Column(Text, nullable=False)
    telegram_message_id = Column(BigInteger, nullable=True)
    is_system = Column(Boolean, default=False, nullable=False)  # Системное сообщение
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Связи
    ticket = relationship("Ticket", back_populates="messages")


# Отдельная база для знаний
KnowledgeBase = declarative_base()


class KnowledgeEntry(KnowledgeBase):
    """Запись в базе знаний"""
    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
