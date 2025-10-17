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

from app import auth
from app.bots import create_dispatcher, start_bot, create_vk_bot, start_vk_bot
from app.config import load_rag_config, load_app_config
from app.db import (
    models,
    tickets_crud as crud,
    get_tickets_session,
    get_knowledge_session,
    init_db,
    TicketsSessionLocal,
    KnowledgeSessionLocal,
    get_session,
    TicketRead,
    KnowledgeStats,
    MessageCreate,
    MessageRead,
    ConversationRead,
)
from app.rag import RAGService, KnowledgeBase
from app.services import ConnectionManager, SimulatorService
from app.auth import require_permission, require_admin, get_user_permissions

logger = logging.getLogger(__name__)

connection_manager = ConnectionManager()
knowledge_base: KnowledgeBase | None = None

templates = Jinja2Templates(directory="app/templates")


def _serialize_tickets(tickets: list[models.Ticket]) -> list[dict]:
    result = []
    for ticket in tickets:
        # Ensure telegram_chat_id is a string when it contains non-numeric VK ids like 'vk_123'
        try:
            # Try direct conversion first
            ticket_data = TicketRead.from_orm(ticket).model_dump(mode="json")
        except Exception:
            # Fallback: coerce telegram_chat_id to string and build dict manually
            try:
                raw = ticket.__dict__.copy()
                raw["telegram_chat_id"] = str(getattr(ticket, "telegram_chat_id", ""))
                # Build minimal payload matching TicketRead fields
                ticket_data = {
                    "id": raw.get("id"),
                    "telegram_chat_id": raw.get("telegram_chat_id"),
                    "title": raw.get("title"),
                    "summary": raw.get("summary"),
                    "status": raw.get("status"),
                    "priority": raw.get("priority"),
                    "created_at": raw.get("created_at"),
                    "first_response_at": raw.get("first_response_at"),
                    "closed_at": raw.get("closed_at"),
                    "updated_at": raw.get("updated_at"),
                }
            except Exception:
                # As last resort, skip this ticket
                logger.exception(
                    "Failed to serialize ticket %s", getattr(ticket, "id", "<unknown>")
                )
                continue
        # Подсчитываем непрочитанные сообщения от user и bot
        unread_count = sum(
            1
            for msg in ticket.messages
            if msg.sender in ["user", "bot"] and not msg.is_read
        )
        # Ограничиваем значение 99
        ticket_data["unread_count"] = min(unread_count, 99)
        result.append(ticket_data)
    return result


def _serialize_message(message: models.Message) -> dict:
    return MessageRead.from_orm(message).model_dump(mode="json")


async def _broadcast_conversations_update(
    session: AsyncSession, manager: ConnectionManager
) -> None:
    """Обновить список заявок для всех подключенных клиентов"""
    tickets = await crud.list_tickets(session, archived=False)
    await manager.broadcast_conversations(_serialize_tickets(tickets))


async def update_popularity_scores():
    """Фоновая задача для обновления популярности вопросов каждые 5 минут"""
    from sqlalchemy import select, text
    from app.db.models import KnowledgeEntry, Message
    from datetime import datetime, timedelta
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # Получаем embedder из конфига
    embedder = None

    while True:
        try:
            await asyncio.sleep(300)  # 5 минут

            logger.info("Обновление счетчиков популярности запросов...")

            # Инициализируем embedder при первом запуске
            if embedder is None:
                rag_config = load_rag_config()
                embedding_cfg = rag_config.get("embeddings", {})
                model_name = embedding_cfg.get(
                    "model_name", "ai-forever/sbert_large_nlu_ru"
                )
                device = embedding_cfg.get("device", "cpu")
                embedder = SentenceTransformer(model_name, device=device)
                logger.info(f"Embedder инициализирован: {model_name}")

            async with KnowledgeSessionLocal() as k_session:
                async with TicketsSessionLocal() as t_session:
                    # Получаем все сообщения за последние 24 часа
                    yesterday = datetime.utcnow() - timedelta(hours=24)

                    # Получаем все вопросы пользователей (не системные)
                    stmt = select(Message).where(
                        Message.created_at >= yesterday,
                        Message.is_system == False,
                        Message.sender == "user",
                    )
                    result = await t_session.execute(stmt)
                    recent_messages = result.scalars().all()

                    # Получаем все записи базы знаний
                    kb_stmt = select(KnowledgeEntry)
                    kb_result = await k_session.execute(kb_stmt)
                    kb_entries = kb_result.scalars().all()

                    if not kb_entries:
                        logger.info("Нет записей в базе знаний")
                        continue

                    if not recent_messages:
                        logger.info(
                            "Нет пользовательских сообщений за последние 24 часа"
                        )
                        # Сбрасываем все счетчики на 0
                        await k_session.execute(
                            text("UPDATE knowledge_entries SET popularity_score = 0.0")
                        )
                        await k_session.commit()
                        continue

                    # Получаем текст всех пользовательских сообщений
                    # Модель Message использует поле `text`, не `content`
                    user_queries = [
                        msg.text
                        for msg in recent_messages
                        if getattr(msg, "text", None)
                    ]
                    if not user_queries:
                        logger.info("Нет текстовых сообщений от пользователей")
                        continue

                    # Вычисляем embeddings для пользовательских запросов
                    logger.info(
                        f"Вычисление embeddings для {len(user_queries)} запросов..."
                    )
                    query_embeddings = await asyncio.to_thread(
                        embedder.encode, user_queries, convert_to_numpy=True
                    )

                    # Подсчитываем популярность каждого вопроса на основе similarity
                    question_scores = {}
                    for entry in kb_entries:
                        if not entry.embedding:
                            # Если нет embedding, вычисляем его
                            entry_embedding = await asyncio.to_thread(
                                embedder.encode, entry.question, convert_to_numpy=True
                            )
                        else:
                            # Используем существующий embedding
                            entry_embedding = np.frombuffer(
                                entry.embedding, dtype=np.float32
                            )

                        # Вычисляем cosine similarity с каждым запросом пользователя
                        similarities = []
                        for query_emb in query_embeddings:
                            # Cosine similarity
                            similarity = np.dot(entry_embedding, query_emb) / (
                                np.linalg.norm(entry_embedding)
                                * np.linalg.norm(query_emb)
                            )
                            similarities.append(similarity)

                        # Считаем среднее значение среди similarity > 0.7 (приблизительно)
                        relevant_count = sum(1 for sim in similarities if sim > 0.7)
                        question_scores[entry.id] = relevant_count

                    # Нормализуем значения в диапазоне 0-1
                    max_count = max(question_scores.values()) if question_scores else 1
                    if max_count == 0:
                        max_count = 1

                    # Обновляем базу данных
                    for entry_id, count in question_scores.items():
                        normalized_score = count / max_count
                        await k_session.execute(
                            text(
                                "UPDATE knowledge_entries SET popularity_score = :score WHERE id = :id"
                            ),
                            {"score": normalized_score, "id": entry_id},
                        )

                    await k_session.commit()
                    logger.info(
                        f"✓ Обновлено {len(question_scores)} записей. Max count: {max_count}"
                    )

                    # Выводим ТОП-3 для отладки
                    top_entries = sorted(
                        question_scores.items(), key=lambda x: x[1], reverse=True
                    )[:3]
                    for entry_id, count in top_entries:
                        entry = next((e for e in kb_entries if e.id == entry_id), None)
                        if entry:
                            logger.info(
                                f"  ТОП: '{entry.question[:50]}...' - {count} релевантных запросов"
                            )

        except asyncio.CancelledError:
            logger.info("Задача обновления счетчиков остановлена")
            break
        except Exception as e:
            logger.exception(f"Ошибка при обновлении счетчиков: {e}")


async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("🚀 Starting application lifespan...")
    await init_db()
    app.state.connection_manager = connection_manager

    # Загружаем полный конфиг (RAG + App)
    from app.config import load_config

    full_config = load_config()

    rag_config = load_rag_config()
    embeddings_cfg = rag_config.get("embeddings", {})

    global knowledge_base
    knowledge_base = KnowledgeBase(
        KnowledgeSessionLocal,
        model_name=embeddings_cfg.get("model_name"),
    )
    await knowledge_base.ensure_loaded()
    app.state.knowledge_base = knowledge_base

    # Используем полный конфиг для RAGService (с настройками speech)
    rag_service = RAGService(full_config)
    await rag_service.prepare()
    app.state.rag = rag_service

    # Инициализируем симулятор
    simulator_service = SimulatorService(rag_service)
    app.state.simulator = simulator_service

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    bot: Bot | None = None
    dispatcher = None
    bot_task: asyncio.Task | None = None
    popularity_task: asyncio.Task | None = None

    if token:
        bot = Bot(token=token)
        # Pass knowledge_base to dispatcher so Telegram bot can query top-k KB entries
        dispatcher = create_dispatcher(
            TicketsSessionLocal, connection_manager, rag_service, knowledge_base
        )
        app.state.bot = bot
        app.state.dispatcher = dispatcher
        bot_task = asyncio.create_task(start_bot(bot, dispatcher))
    else:
        logger.warning(
            "TELEGRAM_BOT_TOKEN is not set. Telegram integration is disabled."
        )
        app.state.bot = None
        app.state.dispatcher = None

    # VK Bot
    vk_token = os.getenv("VK_ACCESS_TOKEN")
    vk_bot_task: asyncio.Task | None = None
    logger.info(f"VK: Token found: {'YES' if vk_token else 'NO'}")
    if vk_token:
        logger.info("VK: Attempting to create VK bot...")
        vk_run_bot = create_vk_bot(
            TicketsSessionLocal, connection_manager, rag_service, vk_token
        )
        if vk_run_bot is not None:
            logger.info("VK: Bot created successfully, starting task...")
            app.state.vk_bot = vk_run_bot
            vk_bot_task = asyncio.create_task(start_vk_bot(vk_run_bot))
            logger.info("VK: Bot task started")
        else:
            logger.warning("VK: Bot disabled due to configuration issues")
            app.state.vk_bot = None
    else:
        logger.warning("VK_ACCESS_TOKEN is not set. VK integration is disabled.")
        app.state.vk_bot = None

    # Запускаем фоновую задачу обновления популярности
    popularity_task = asyncio.create_task(update_popularity_scores())
    logger.info("✓ Фоновая задача обновления популярности запущена")

    try:
        yield
    finally:
        if bot_task:
            bot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await bot_task
        if vk_bot_task:
            vk_bot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await vk_bot_task
        if popularity_task:
            popularity_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await popularity_task
        if bot:
            await bot.session.close()
        await connection_manager.close_all()


app = FastAPI(title="Support Desk", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint для Docker"""
    return {"status": "ok", "service": "Support Desk"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница - перенаправляет после аутентификации пользователей"""
    if not auth.is_authenticated_request(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/tickets", response_class=HTMLResponse)
@require_permission("tickets")
async def tickets_page(request: Request):
    """Страница заявок"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/tickets/{ticket_id}", response_class=HTMLResponse)
@require_permission("tickets")
async def ticket_detail_page(request: Request, ticket_id: int):
    """Страница конкретной заявки - возвращает тот же HTML, JavaScript загрузит нужные данные"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if auth.is_authenticated_request(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request):
    data = await request.json()
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    user = auth.validate_credentials(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

    response = JSONResponse({"success": True})
    auth.issue_session_cookie(response, user["id"])
    return response


@app.post("/logout")
async def logout(request: Request):
    accept = request.headers.get("accept", "") or ""
    if "application/json" in accept and "text/html" not in accept:
        response = JSONResponse({"success": True})
    else:
        response = RedirectResponse(url="/login", status_code=303)

    # Очищаем cookie сессии
    auth.clear_session_cookie(response)

    # Добавляем заголовки для очистки кеша
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response


@app.get("/api/me")
async def get_current_user(request: Request):
    """Получить информацию о текущем пользователе"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    from app.auth import user_manager

    user_id = auth.get_user_id_from_request(request)
    user = user_manager.get_user_by_id(user_id)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {"username": user["username"], "authenticated": True}


@app.get("/api/permissions")
async def get_user_permissions_api(request: Request):
    """Получить права текущего активного пользователя"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Not authenticated")

    permissions = get_user_permissions(request)
    return {"available_pages": permissions}


@app.get("/admin/knowledge", response_class=HTMLResponse)
@require_permission("knowledge")
async def knowledge_admin(request: Request):
    async with KnowledgeSessionLocal() as session:
        total = await crud.count_knowledge_entries(session)
    return templates.TemplateResponse(
        "knowledge.html",
        {"request": request, "entry_count": total},
    )


@app.get("/dashboard", response_class=HTMLResponse)
@require_permission("dashboard")
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/simulator", response_class=HTMLResponse)
@require_permission("simulator")
async def simulator(request: Request):
    return templates.TemplateResponse("simulator.html", {"request": request})


# ==================== FAQ (PUBLIC) ====================


@app.get("/faq", response_class=HTMLResponse)
async def faq_page(request: Request):
    """Публичная страница FAQ для пользователей"""
    return templates.TemplateResponse("faq.html", {"request": request})


@app.get("/api/faq")
async def get_faq(session: AsyncSession = Depends(get_knowledge_session)):
    """API для получения всех вопросов отсортированных по популярности"""
    from sqlalchemy import select, desc
    from app.db.models import KnowledgeEntry

    # Получаем все записи, отсортированные по популярности
    stmt = select(KnowledgeEntry).order_by(desc(KnowledgeEntry.popularity_score))
    result = await session.execute(stmt)
    entries = result.scalars().all()

    # Формируем ответ
    items = []
    for entry in entries:
        items.append(
            {
                "id": entry.id,
                "question": entry.question,
                "answer": entry.answer,
                "popularity_score": entry.popularity_score,
            }
        )

    return {"items": items}


@app.get("/api/faq/search")
async def search_faq(q: str, session: AsyncSession = Depends(get_knowledge_session)):
    """Поиск вопросов в FAQ по векторным embeddings"""
    from sqlalchemy import select
    from app.db.models import KnowledgeEntry
    import numpy as np

    if not q or len(q.strip()) < 2:
        return {"items": []}

    query_text = q.strip()

    # Получаем knowledge_base из app state
    knowledge_base = app.state.knowledge_base

    try:
        # Получаем embedding для запроса
        query_embedding = await asyncio.to_thread(
            knowledge_base.model.encode, query_text, convert_to_numpy=True
        )

        # Получаем все записи
        stmt = select(KnowledgeEntry)
        result = await session.execute(stmt)
        entries = result.scalars().all()

        if not entries:
            return {"items": []}

        # Вычисляем similarity для каждой записи
        results = []
        for entry in entries:
            if entry.embedding:
                entry_embedding = np.frombuffer(entry.embedding, dtype=np.float32)

                # Cosine similarity
                similarity = np.dot(query_embedding, entry_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry_embedding)
                )

                # Добавляем только релевантные результаты (similarity > 0.3)
                if similarity > 0.3:
                    results.append(
                        {
                            "id": entry.id,
                            "question": entry.question,
                            "answer": entry.answer,
                            "popularity_score": entry.popularity_score,
                            "similarity": float(similarity),
                        }
                    )

        # Сортируем по similarity (от большего к меньшему)
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Возвращаем топ-20 результатов
        return {"items": results[:20]}

    except Exception as e:
        logger.exception(f"Search error: {e}")
        # Fallback на простой текстовый поиск
        stmt = select(KnowledgeEntry)
        result = await session.execute(stmt)
        entries = result.scalars().all()

        query_lower = query_text.lower()
        filtered = [
            {
                "id": e.id,
                "question": e.question,
                "answer": e.answer,
                "popularity_score": e.popularity_score,
                "similarity": 0.5,
            }
            for e in entries
            if query_lower in e.question.lower() or query_lower in e.answer.lower()
        ]

        return {"items": filtered[:20]}


@app.get("/api/dashboard/stats")
async def get_dashboard_stats(
    request: Request,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
):
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    # Получаем статистику
    tickets_stats = await crud.get_tickets_stats(session)
    response_time_stats = await crud.get_response_time_stats(session)
    daily_stats = await crud.get_daily_tickets_stats(session, days=30)
    daily_time_metrics = await crud.get_daily_time_metrics(session, days=30)

    # Метрики времени
    avg_response_time = await crud.get_average_response_time(session)
    avg_resolution_time = await crud.get_average_resolution_time(session)

    # Статистика по базе знаний
    async with KnowledgeSessionLocal() as knowledge_session:
        knowledge_count = await crud.count_knowledge_entries(knowledge_session)

    return {
        "tickets": tickets_stats,
        "response_times": response_time_stats,
        "daily_tickets": daily_stats,
        "daily_time_metrics": daily_time_metrics,
        "knowledge_entries": knowledge_count,
        "avg_response_time_minutes": avg_response_time,
        "avg_resolution_time_minutes": avg_resolution_time,
    }


@app.post("/admin/knowledge/upload")
async def upload_knowledge(
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_knowledge_session),
    _: None = Depends(auth.ensure_api_auth),
) -> JSONResponse:
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.exception("Pandas import failed: %s", exc)
        raise HTTPException(
            status_code=500, detail="Pandas is required on the server"
        ) from exc

    try:
        dataframe = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to read Excel: {exc}"
        ) from exc

    if dataframe.empty:
        raise HTTPException(status_code=400, detail="File contains no records")

    normalized = {str(col).strip().lower(): col for col in dataframe.columns}
    question_column = next(
        (
            normalized[key]
            for key in ("question", "������", "questions", "�������")
            if key in normalized
        ),
        None,
    )
    answer_column = next(
        (
            normalized[key]
            for key in ("answer", "�����", "answers", "������")
            if key in normalized
        ),
        None,
    )

    if question_column is None or answer_column is None:
        if len(dataframe.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least two columns with questions and answers are required",
            )
        question_column = dataframe.columns[0]
        answer_column = dataframe.columns[1]

    pairs: list[tuple[str, str]] = []
    for _, row in dataframe.iterrows():
        question_cell = row[question_column]
        answer_cell = row[answer_column]
        if pd.isna(question_cell) or pd.isna(answer_cell):
            continue
        question = str(question_cell).strip()
        answer = str(answer_cell).strip()
        if not question or not answer:
            continue
        pairs.append((question, answer))

    if not pairs:
        raise HTTPException(
            status_code=400, detail="No valid question-answer pairs found"
        )

    kb: KnowledgeBase = request.app.state.knowledge_base
    entries_count = await kb.rebuild_from_pairs(pairs)
    await request.app.state.rag.reload()
    return JSONResponse({"success": True, "entries": entries_count})


@app.get("/api/knowledge/stats", response_model=KnowledgeStats)
async def knowledge_stats(
    session: AsyncSession = Depends(get_knowledge_session),
    _: None = Depends(auth.ensure_api_auth),
) -> KnowledgeStats:
    total = await crud.count_knowledge_entries(session)
    return KnowledgeStats(total_entries=total)


@app.get("/api/conversations", response_model=list[TicketRead])
async def api_list_conversations(
    archived: bool = False,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> list[TicketRead]:
    tickets = await crud.list_tickets(session, archived=archived)
    return tickets


@app.get("/api/conversations/{conversation_id}", response_model=ConversationRead)
async def api_get_conversation(
    conversation_id: int,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> ConversationRead:
    ticket = await crud.get_ticket_by_id(session, conversation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    # Возвращаем полные данные о тикете
    return TicketRead.from_orm(ticket)


@app.get(
    "/api/conversations/{conversation_id}/messages", response_model=list[MessageRead]
)
async def api_list_messages(
    conversation_id: int,
    include_system: bool = False,  # Включать системные сообщения
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> list[MessageRead]:
    ticket = await crud.get_ticket_by_id(session, conversation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    messages = await crud.list_messages_for_ticket(
        session, conversation_id, include_system=include_system
    )
    return messages


@app.post("/api/conversations/{conversation_id}/finish")
async def api_finish(
    conversation_id: int,
    request: Request,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> JSONResponse:
    ticket = await crud.get_ticket_by_id(session, conversation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if ticket.status in [models.TicketStatus.CLOSED, models.TicketStatus.ARCHIVED]:
        raise HTTPException(status_code=400, detail="Ticket already closed")

    finish_text = 'Оператор завершил заявку. Если потребуется помощь, напишите снова "Позови оператора".'

    bot: Bot | None = request.app.state.bot
    # Send finish notification to the correct platform
    if isinstance(ticket.telegram_chat_id, str) and ticket.telegram_chat_id.startswith(
        "vk_"
    ):
        try:
            from app.bots.vk_bot import VK_API

            if VK_API is not None:
                peer = int(ticket.telegram_chat_id.split("_", 1)[1])
                try:
                    import vk_api

                    try:
                        random_id = vk_api.utils.get_random_id()
                    except Exception:
                        random_id = 0
                except Exception:
                    random_id = 0
                VK_API.messages.send(
                    peer_id=peer, message=finish_text, random_id=random_id
                )
            else:
                logger.warning(
                    "VK API client not initialized; cannot send finish notification"
                )
        except Exception as e:
            logger.exception(f"Failed to send finish notification via VK: {e}")
    else:
        if bot is not None:
            await bot.send_message(ticket.telegram_chat_id, finish_text)

    # Добавляем финальное сообщение и закрываем заявку
    finish_message = await crud.add_message(
        session, conversation_id, "bot", finish_text, is_system=True
    )
    await crud.update_ticket_status(
        session, conversation_id, models.TicketStatus.CLOSED
    )

    # Отмечаем закрытие заявки в RAG сервисе для правильной сегментации истории
    rag_service: RAGService = request.app.state.rag
    rag_service.mark_ticket_closed(ticket.telegram_chat_id)

    manager: ConnectionManager = request.app.state.connection_manager
    await manager.broadcast_message(conversation_id, _serialize_message(finish_message))
    tickets = await crud.list_tickets(session, archived=False)
    await manager.broadcast_conversations(_serialize_tickets(tickets))

    return JSONResponse({"success": True})


@app.post("/api/conversations/{conversation_id}/reply", response_model=MessageRead)
async def api_reply(
    conversation_id: int,
    message: MessageCreate,
    request: Request,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> MessageRead:
    ticket = await crud.get_ticket_by_id(session, conversation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    if ticket.status in [models.TicketStatus.CLOSED, models.TicketStatus.ARCHIVED]:
        raise HTTPException(status_code=400, detail="Cannot reply to closed ticket")

    # Переводим заявку в статус "В работе" если она была открыта
    if ticket.status == models.TicketStatus.OPEN:
        await crud.update_ticket_status(
            session, conversation_id, models.TicketStatus.IN_PROGRESS
        )

    # Формируем ответ для оператора
    await crud.set_first_response_time(session, conversation_id)

    # Получаем текущего пользователя из запроса
    from app.auth import get_current_user_from_request

    current_user = get_current_user_from_request(request)
    operator_name = (
        current_user.get("full_name", "Неизвестный") if current_user else "Неизвестный"
    )

    # Форматируем сообщение для оператора
    formatted_message = f"<b>Оператор {operator_name}:</b>\n{message.text}"

    bot: Bot | None = request.app.state.bot
    # If ticket.telegram_chat_id is a VK chat id (vk_...), send via VK instead of Telegram
    if isinstance(ticket.telegram_chat_id, str) and ticket.telegram_chat_id.startswith(
        "vk_"
    ):
        try:
            from app.bots.vk_bot import VK_API

            if VK_API is not None:
                # peer_id is the numeric part after 'vk_'
                peer = int(ticket.telegram_chat_id.split("_", 1)[1])
                try:
                    import vk_api

                    try:
                        random_id = vk_api.utils.get_random_id()
                    except Exception:
                        random_id = 0
                except Exception:
                    random_id = 0
                VK_API.messages.send(
                    peer_id=peer, message=formatted_message, random_id=random_id
                )
            else:
                logger.warning(
                    "VK API client not initialized; cannot send operator message"
                )
        except Exception as e:
            logger.exception(f"Failed to send operator reply via VK: {e}")
    else:
        if bot is not None:
            await bot.send_message(
                ticket.telegram_chat_id, formatted_message, parse_mode="HTML"
            )

    new_message = await crud.add_message(
        session, conversation_id, "operator", message.text, is_system=False
    )

    manager: ConnectionManager = request.app.state.connection_manager
    await manager.broadcast_message(conversation_id, _serialize_message(new_message))

    # Отправляем обновление списка заявок
    await _broadcast_conversations_update(session, manager)

    return new_message


@app.post("/api/conversations/{conversation_id}/read")
async def mark_conversation_read(
    conversation_id: int,
    request: Request,
    _: None = Depends(auth.ensure_api_auth),
) -> dict:
    """Отметить разговор как прочитанный (для пользователя)"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    return {"success": True}


@app.get("/api/conversations/{conversation_id}/summary")
async def get_ticket_summary(
    conversation_id: int,
    request: Request,
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> dict:
    """Получить сводку по заявке"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")

    # Получаем заявку и связанные с ней сообщения
    ticket_with_messages = await crud.get_ticket_with_messages(session, conversation_id)
    if ticket_with_messages is None:
        raise HTTPException(status_code=404, detail="Ticket not found")

    ticket, messages = ticket_with_messages

    # Получаем summary из сообщения, если оно есть
    if ticket.summary:
        summary = ticket.summary
    else:
        # Генерируем summary из сообщений
        rag_service: RAGService = request.app.state.rag
        summary = await rag_service.generate_ticket_summary(
            messages, ticket_id=conversation_id
        )
        await crud.update_ticket_summary(session, conversation_id, summary)

    return {
        "ticket_id": conversation_id,
        "summary": summary,
        "status": ticket.status,
        "created_at": ticket.created_at.isoformat(),
        "message_count": len(messages),
    }


@app.websocket("/ws/conversations")
async def websocket_conversations(websocket: WebSocket) -> None:
    if not auth.is_authenticated_websocket(websocket):
        await websocket.close(code=auth.WEBSOCKET_UNAUTHORIZED_CLOSE_CODE)
        return
    await websocket.accept()
    await connection_manager.register_conversations(websocket)
    try:
        async with TicketsSessionLocal() as session:
            tickets = await crud.list_tickets(session, archived=False)
            await connection_manager.send_conversations_snapshot(
                websocket, _serialize_tickets(tickets)
            )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.unregister_conversations(websocket)


@app.websocket("/ws/conversations/{conversation_id}")
async def websocket_messages(websocket: WebSocket, conversation_id: int) -> None:
    if not auth.is_authenticated_websocket(websocket):
        await websocket.close(code=auth.WEBSOCKET_UNAUTHORIZED_CLOSE_CODE)
        return

    async with TicketsSessionLocal() as session:
        ticket = await crud.get_ticket_by_id(session, conversation_id)
        if ticket is None:
            await websocket.close(code=4404)
            return
        messages = await crud.list_messages_for_ticket(
            session, conversation_id, include_system=False
        )
        history_payload = [_serialize_message(item) for item in messages]

    # Принимаем WebSocket соединение и регистрируем чат
    await websocket.accept()
    await connection_manager.register_chat(conversation_id, websocket)

    try:
        # Ждем, пока пользователь не отправит сообщение (пока has_active_chat_connections будет True)
        async with TicketsSessionLocal() as session:
            marked_count = await crud.mark_ticket_messages_as_read(
                session, conversation_id
            )
            if marked_count > 0:
                # Отправляем обновление списка заявок
                await _broadcast_conversations_update(session, connection_manager)

        await connection_manager.send_message_history(
            websocket, conversation_id, history_payload
        )
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await connection_manager.unregister_chat(conversation_id, websocket)
        if websocket.client_state.name == "CONNECTED":
            await websocket.close()


# ==================== Simulator API ====================


@app.get("/api/simulator/characters")
async def get_simulator_characters(request: Request):
    """Получить список доступных персонажей для симуляции"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    simulator: SimulatorService = request.app.state.simulator
    return {"characters": simulator.characters}


@app.post("/api/simulator/start")
async def start_simulator_session(request: Request, data: dict = Body(...)):
    """Запустить новую сессию симуляции"""
    try:
        if not auth.is_authenticated_request(request):
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Извлекаем параметры из body
        character = data.get("character", "medium")

        # Получаем session cookie для user_id
        settings = auth.get_settings()
        user_id = request.cookies.get(settings.cookie_name, "anonymous")
        simulator: SimulatorService = request.app.state.simulator

        # Запускаем сессию
        session = simulator.start_session(user_id, character)

        # Генерируем первый вопрос
        question = await simulator.generate_question(session)

        return {
            "session_id": user_id,
            "character": session.character,
            "character_info": simulator.characters.get(session.character, {}),
            "questions_total": session.questions_count,
            "current_question": session.current_question + 1,
            "question": question.question,
        }
    except Exception as e:
        logger.error(f"Failed to start simulator session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"������ ������� ������: {str(e)}")


@app.post("/api/simulator/respond")
async def respond_to_question(request: Request):
    """Отправить ответ на вопрос"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    settings = auth.get_settings()
    user_id = request.cookies.get(settings.cookie_name, "anonymous")
    simulator: SimulatorService = request.app.state.simulator

    data = await request.json()
    user_answer = data.get("answer", "")

    session = simulator.get_session(user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Оцениваем ответ
    evaluation = simulator.evaluate_response(session, user_answer)

    # Добавляем ответ
    session.add_response(user_answer, evaluation)

    # Проверяем, завершена ли сессия
    is_complete = session.is_complete()

    response_data = {
        "score": evaluation.score,
        "feedback": evaluation.feedback,
        "ai_suggestion": evaluation.ai_suggestion,
        "is_correct": evaluation.is_correct,
        "session_complete": is_complete,
        "current_question": session.current_question,
        "questions_total": session.questions_count,
    }

    # Если сессия не завершена - генерируем следующий вопрос
    if not is_complete:
        next_question = await simulator.generate_question(session)
        response_data["next_question"] = next_question.question
    else:
        # Сессия завершена - подводим итоги
        response_data["total_score"] = session.total_score
        response_data["average_score"] = session.get_average_score()
        response_data["history"] = session.history

    return response_data


@app.get("/api/simulator/hint")
async def get_hint(request: Request):
    """Получить подсказку для текущего вопроса"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    settings = auth.get_settings()
    user_id = request.cookies.get(settings.cookie_name, "anonymous")
    simulator: SimulatorService = request.app.state.simulator

    session = simulator.get_session(user_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    hint = simulator.get_hint(session)

    return {"hint": hint}


@app.post("/api/simulator/end")
async def end_simulator_session(request: Request):
    """Завершить сессию симуляции"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    settings = auth.get_settings()
    user_id = request.cookies.get(settings.cookie_name, "anonymous")
    simulator: SimulatorService = request.app.state.simulator

    session = simulator.get_session(user_id)
    if session:
        final_stats = {
            "questions_answered": len(session.history),
            "total_score": session.total_score,
            "average_score": session.get_average_score(),
            "history": session.history,
        }
        simulator.end_session(user_id)
        return {"message": "Session ended", "stats": final_stats}

    return {"message": "No active session"}


# ==================== ADMIN: USER MANAGEMENT ====================


@app.get("/admin/users", response_class=HTMLResponse)
@require_admin()
async def users_admin(request: Request):
    """Получить список пользователей для администрирования"""
    return templates.TemplateResponse("admin_users.html", {"request": request})


@app.get("/api/admin/users")
@require_admin(redirect_to_home=False)
async def get_users(request: Request):
    """Получить список всех пользователей"""
    from app.auth import user_manager

    users = user_manager.get_all_users()

    # Удаляем password_hash из ответа
    for user in users:
        user.pop("password_hash", None)

    return {"users": users}


@app.get("/api/user/permissions")
async def get_current_user_permissions(request: Request):
    """Получить права доступа текущего пользователя"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")

    permissions = get_user_permissions(request)
    return {"permissions": permissions}


@app.post("/api/admin/users")
@require_admin(redirect_to_home=False)
async def create_user(request: Request, data: dict = Body(...)):
    """Создать нового пользователя"""

    from app.auth import user_manager

    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    full_name = data.get("full_name", "").strip()
    role_id = data.get("role_id")
    is_admin = data.get("is_admin", False)

    if not username or not password or not full_name or not role_id:
        raise HTTPException(status_code=400, detail="��� ���� �����������")

    user_id = user_manager.create_user(username, password, full_name, role_id, is_admin)

    if user_id is None:
        raise HTTPException(
            status_code=400, detail="Пользователь с такими данными уже существует"
        )

    return {"success": True, "user_id": user_id}


@app.put("/api/admin/users/{user_id}")
@require_admin(redirect_to_home=False)
async def update_user(request: Request, user_id: int, data: dict = Body(...)):
    """Обновить данные пользователя"""
    # Проверяем, что нельзя обновить суперпользователя
    if user_id == 1:
        raise HTTPException(
            status_code=403, detail="Изменение суперпользователя запрещено"
        )

    from app.auth import user_manager

    success = user_manager.update_user(user_id, **data)

    if not success:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {"success": True}


@app.delete("/api/admin/users/{user_id}")
@require_admin(redirect_to_home=False)
async def delete_user(request: Request, user_id: int):
    """Удалить пользователя"""
    # Проверяем, что нельзя удалить суперпользователя
    if user_id == 1:
        raise HTTPException(
            status_code=403, detail="Удаление суперпользователя запрещено"
        )

    from app.auth import user_manager

    success = user_manager.delete_user(user_id)

    if not success:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {"success": True}


@app.post("/api/admin/users/{user_id}/password")
@require_admin(redirect_to_home=False)
async def change_password(request: Request, user_id: int, data: dict = Body(...)):
    from app.auth import user_manager

    new_password = data.get("password", "").strip()

    if not new_password:
        raise HTTPException(status_code=400, detail="Новый пароль не может быть пустым")

    success = user_manager.update_password(user_id, new_password)

    if not success:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {"success": True}


# ==================== ADMIN: PERMISSIONS ====================


@app.get("/api/admin/permissions")
@require_admin(redirect_to_home=False)
async def get_permissions(request: Request):

    from app.auth import user_manager

    permissions = user_manager.get_all_permissions()

    return {"permissions": permissions}


@app.get("/api/admin/users/{user_id}/permissions")
@require_admin(redirect_to_home=False)
async def get_user_permissions_admin(request: Request, user_id: int):
    from app.auth import user_manager

    permissions = user_manager.get_user_permissions(user_id)

    return {"permissions": permissions}


@app.post("/api/admin/users/{user_id}/permissions")
@require_admin(redirect_to_home=False)
async def set_user_permission(request: Request, user_id: int, data: dict = Body(...)):
    from app.auth import user_manager

    page_key = data.get("page_key")
    granted = data.get("granted", True)

    if not page_key:
        raise HTTPException(status_code=400, detail="page_key не указанный")

    success = user_manager.set_user_permission(user_id, page_key, granted)

    if not success:
        raise HTTPException(status_code=404, detail="Пользователь не найден")

    return {"success": True}


# ==================== ADMIN: ROLES ====================


@app.get("/api/admin/roles")
@require_admin(redirect_to_home=False)
async def get_roles(request: Request):

    from app.auth import user_manager

    roles = user_manager.get_all_roles()

    return {"roles": roles}


@app.get("/api/admin/roles/{role_id}/permissions")
@require_admin(redirect_to_home=False)
async def get_role_permissions(request: Request, role_id: int):
    from app.auth import user_manager

    permissions = user_manager.get_role_permissions(role_id)

    return {"permissions": permissions}


@app.post("/api/admin/roles/{role_id}/permissions")
@require_admin(redirect_to_home=False)
async def set_role_permissions(request: Request, role_id: int, data: dict = Body(...)):
    from app.auth import user_manager

    page_keys = data.get("page_keys", [])

    success = user_manager.set_role_permissions(role_id, page_keys)

    return {"success": success}


# ==================== ADMIN: SETTINGS ====================


@app.get("/api/admin/settings")
@require_admin(redirect_to_home=False)
async def get_settings_admin(request: Request):
    from app.auth import user_manager

    settings = user_manager.get_all_settings()

    return {"settings": settings}


@app.post("/api/admin/settings")
@require_admin(redirect_to_home=False)
async def update_setting(request: Request, data: dict = Body(...)):
    from app.auth import user_manager

    key = data.get("key")
    value = data.get("value")

    if not key or value is None:
        raise HTTPException(status_code=400, detail="key или value не указаны")

    success = user_manager.set_setting(key, str(value))

    if not success:
        raise HTTPException(status_code=404, detail="Настройка не найдена")

    return {"success": True}
