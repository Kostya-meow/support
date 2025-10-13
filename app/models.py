from __future__ import annotations

from datetime import datetime
from enum import Enum

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
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
    telegram_chat_id = Column(String(255), index=True, nullable=False)  # Изменено на String для поддержки VK
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)  # Краткое описание заявки (авто-генерируется)
    status = Column(String(20), default=TicketStatus.OPEN, nullable=False)
    priority = Column(String(10), default="medium", nullable=False)  # low, medium, high
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    first_response_at = Column(DateTime, nullable=True)  # Время первого ответа оператора
    closed_at = Column(DateTime, nullable=True)  # Время закрытия заявки
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

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
    vk_message_id = Column(BigInteger, nullable=True)
    is_system = Column(Boolean, default=False, nullable=False)  # Системное сообщение
    is_read = Column(Boolean, default=False, nullable=False)  # Прочитано оператором
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
    popularity_score = Column(Float, default=0.0, nullable=False)
