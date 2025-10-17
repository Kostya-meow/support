from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
from typing import AsyncIterator

from dotenv import load_dotenv

load_dotenv()

from aiogram import Bot
from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    Body,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app import models, tickets_crud as crud
from app import auth
from app.bots import create_dispatcher, start_bot, create_vk_bot, start_vk_bot
from app.config import load_rag_config, load_app_config
from app.database import (
    get_tickets_session,
    get_knowledge_session,
    init_db,
    TicketsSessionLocal,
    KnowledgeSessionLocal,
    get_session,
)
from app.rag import RAGService, KnowledgeBase
from app.realtime import ConnectionManager
from app.schemas import (
    TicketRead,
    KnowledgeStats,
    MessageCreate,
    MessageRead,
    ConversationRead,
)
from app.simulator_service import SimulatorService
from app.auth import require_permission, require_admin, get_user_permissions

logger = logging.getLogger(__name__)
