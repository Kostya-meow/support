"""Database module - models, schemas, CRUD operations, and connections."""

# Database connections and sessions
from app.db.database import (
    get_tickets_session,
    get_knowledge_session,
    get_session,
    init_db,
    TicketsSessionLocal,
    KnowledgeSessionLocal,
    KnowledgeBase as KnowledgeDBBase,
    TicketsBase,
)

# SQLAlchemy models
from app.db.models import (
    Ticket,
    Message,
    KnowledgeEntry,
    TicketStatus,
)

# Pydantic schemas
from app.db.schemas import (
    TicketRead,
    TicketCreate,
    TicketUpdate,
    MessageRead,
    MessageCreate,
    ConversationRead,
    KnowledgeStats,
)

# CRUD operations
from app.db import crud
from app.db import tickets_crud

__all__ = [
    # Database
    "get_tickets_session",
    "get_knowledge_session",
    "get_session",
    "init_db",
    "TicketsSessionLocal",
    "KnowledgeSessionLocal",
    "KnowledgeDBBase",
    "TicketsBase",
    # Models
    "Ticket",
    "Message",
    "KnowledgeEntry",
    "TicketStatus",
    # Schemas
    "TicketRead",
    "TicketCreate",
    "TicketUpdate",
    "MessageRead",
    "MessageCreate",
    "ConversationRead",
    "KnowledgeStats",
    # CRUD modules
    "crud",
    "tickets_crud",
]
