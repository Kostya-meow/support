from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.models import TicketStatus


class TicketRead(BaseModel):
    id: int
    telegram_chat_id: int
    title: Optional[str]
    summary: Optional[str]
    status: TicketStatus
    priority: str
    created_at: datetime
    first_response_at: Optional[datetime]
    closed_at: Optional[datetime]
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
    
    @property
    def is_archived(self) -> bool:
        """Проверяет, является ли заявка архивной"""
        from app.models import TicketStatus
        return self.status in [TicketStatus.CLOSED, TicketStatus.ARCHIVED]


class MessageRead(BaseModel):
    id: int
    ticket_id: int
    sender: str
    text: str
    is_system: bool = False
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
    
    # Для обратной совместимости с API
    @property
    def conversation_id(self) -> int:
        return self.ticket_id


class MessageCreate(BaseModel):
    text: str


class KnowledgeStats(BaseModel):
    total_entries: int


# Обратная совместимость
class ConversationRead(TicketRead):
    """Для обратной совместимости с API."""
    
    @property
    def operator_requested(self) -> bool:
        return self.status in [TicketStatus.OPEN, TicketStatus.IN_PROGRESS]
    
    @property
    def unread_count(self) -> int:
        return 0  # Пока не реализуем подсчет непрочитанных
    
    @property
    def is_archived(self) -> bool:
        return self.status == TicketStatus.ARCHIVED
