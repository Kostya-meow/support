"""Database module - models, schemas, CRUD operations, and connections."""

# Database connections and sessions
from app.db.database import (
    get_tickets_session,
    get_knowledge_session,
    get_session,
    init_db,
    TicketsSessionLocal,
    KnowledgeSessionLocal,
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
    # Models
    "Ticket",
    "Message",
    "KnowledgeEntry",
    "TicketStatus",
    "models",
    # Schemas
    "TicketRead",
    "MessageRead",
    "MessageCreate",
    "ConversationRead",
    "KnowledgeStats",
    # CRUD modules
    "crud",
    "tickets_crud",
]
