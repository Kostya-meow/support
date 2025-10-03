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
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app import auth, models, tickets_crud as crud
from app.bot import create_dispatcher, start_bot
from app.config import load_config
from app.database import get_tickets_session, get_knowledge_session, init_db, TicketsSessionLocal, KnowledgeSessionLocal, get_session
from app.rag_service import RAGService
from app.realtime import ConnectionManager
from app.retrieval import KnowledgeBase
from app.schemas import TicketRead, KnowledgeStats, MessageCreate, MessageRead, ConversationRead
from app.simulator_service import SimulatorService

logger = logging.getLogger(__name__)

connection_manager = ConnectionManager()
knowledge_base: KnowledgeBase | None = None

templates = Jinja2Templates(directory="app/templates")


def _serialize_tickets(tickets: list[models.Ticket]) -> list[dict]:
    return [TicketRead.from_orm(item).model_dump(mode="json") for item in tickets]


def _serialize_message(message: models.Message) -> dict:
    return MessageRead.from_orm(message).model_dump(mode="json")


async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    app.state.connection_manager = connection_manager

    config = load_config()
    embeddings_cfg = config.get("embeddings", {})

    global knowledge_base
    knowledge_base = KnowledgeBase(
        KnowledgeSessionLocal,
        model_name=embeddings_cfg.get("model_name"),
    )
    await knowledge_base.ensure_loaded()
    app.state.knowledge_base = knowledge_base

    rag_service = RAGService(config)
    await rag_service.prepare()
    app.state.rag = rag_service
    
    # Инициализируем симулятор
    simulator_service = SimulatorService(rag_service)
    app.state.simulator = simulator_service

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    bot: Bot | None = None
    dispatcher = None
    bot_task: asyncio.Task | None = None

    if token:
        bot = Bot(token=token)
        dispatcher = create_dispatcher(TicketsSessionLocal, connection_manager, rag_service)
        app.state.bot = bot
        app.state.dispatcher = dispatcher
        bot_task = asyncio.create_task(start_bot(bot, dispatcher))
    else:
        logger.warning("TELEGRAM_BOT_TOKEN is not set. Telegram integration is disabled.")
        app.state.bot = None
        app.state.dispatcher = None

    try:
        yield
    finally:
        if bot_task:
            bot_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await bot_task
        if bot:
            await bot.session.close()
        await connection_manager.close_all()


app = FastAPI(title="Support Desk", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not auth.is_authenticated_request(request):
        return RedirectResponse(url="/login", status_code=303)
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
    if not auth.validate_credentials(username, password):
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")
    response = JSONResponse({"success": True})
    auth.issue_session_cookie(response)
    return response


@app.post("/logout")
async def logout(request: Request):
    accept = request.headers.get("accept", "") or ""
    if "application/json" in accept and "text/html" not in accept:
        response = JSONResponse({"success": True})
    else:
        response = RedirectResponse(url="/login", status_code=303)
    auth.clear_session_cookie(response)
    return response


@app.get("/admin/knowledge", response_class=HTMLResponse)
async def knowledge_admin(request: Request):
    if not auth.is_authenticated_request(request):
        return RedirectResponse(url="/login", status_code=303)
    async with KnowledgeSessionLocal() as session:
        total = await crud.count_knowledge_entries(session)
    return templates.TemplateResponse(
        "knowledge.html",
        {"request": request, "entry_count": total},
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    if not auth.is_authenticated_request(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/simulator", response_class=HTMLResponse)
async def simulator(request: Request):
    if not auth.is_authenticated_request(request):
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse("simulator.html", {"request": request})


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
        "knowledge_entries": knowledge_count,
        "avg_response_time_minutes": avg_response_time,
        "avg_resolution_time_minutes": avg_resolution_time
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
        raise HTTPException(status_code=500, detail="Pandas is required on the server") from exc

    try:
        dataframe = pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read Excel: {exc}") from exc

    if dataframe.empty:
        raise HTTPException(status_code=400, detail="File contains no records")

    normalized = {str(col).strip().lower(): col for col in dataframe.columns}
    question_column = next(
        (normalized[key] for key in ("question", "вопрос", "questions", "вопросы") if key in normalized),
        None,
    )
    answer_column = next(
        (normalized[key] for key in ("answer", "ответ", "answers", "ответы") if key in normalized),
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
        raise HTTPException(status_code=400, detail="No valid question-answer pairs found")

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


@app.get("/api/conversations/{conversation_id}/messages", response_model=list[MessageRead])
async def api_list_messages(
    conversation_id: int,
    include_system: bool = False,  # По умолчанию скрываем системные сообщения
    session: AsyncSession = Depends(get_tickets_session),
    _: None = Depends(auth.ensure_api_auth),
) -> list[MessageRead]:
    ticket = await crud.get_ticket_by_id(session, conversation_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    messages = await crud.list_messages_for_ticket(session, conversation_id, include_system=include_system)
    return messages


# Временно отключено - в новой архитектуре не используется подсчет непрочитанных
# @app.post("/api/conversations/{conversation_id}/read", response_model=ConversationRead)
# async def api_mark_read(...):
#     pass


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

    finish_text = (
        "Оператор завершил заявку. Если потребуется помощь, напишите снова или нажмите кнопку \"Позвать оператора\"."
    )

    bot: Bot | None = request.app.state.bot
    if bot is not None:
        await bot.send_message(ticket.telegram_chat_id, finish_text)

    # Добавляем финальное сообщение и закрываем заявку
    finish_message = await crud.add_message(session, conversation_id, "bot", finish_text, is_system=True)
    await crud.update_ticket_status(session, conversation_id, models.TicketStatus.CLOSED)
    
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
        await crud.update_ticket_status(session, conversation_id, models.TicketStatus.IN_PROGRESS)
    
    # Записываем время первого ответа оператора (если еще не записано)
    await crud.set_first_response_time(session, conversation_id)

    bot: Bot | None = request.app.state.bot
    if bot is not None:
        await bot.send_message(ticket.telegram_chat_id, message.text)

    new_message = await crud.add_message(session, conversation_id, "operator", message.text, is_system=False)

    manager: ConnectionManager = request.app.state.connection_manager
    await manager.broadcast_message(conversation_id, _serialize_message(new_message))

    return new_message


@app.post("/api/conversations/{conversation_id}/read")
async def mark_conversation_read(
    conversation_id: int,
    request: Request,
    _: None = Depends(auth.ensure_api_auth),
) -> dict:
    """Отметить заявку как прочитанную (заглушка для совместимости)"""
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
    """Получить краткое саммари тикета для оператора"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Получаем тикет с сообщениями через специальную функцию
    ticket_with_messages = await crud.get_ticket_with_messages(session, conversation_id)
    if ticket_with_messages is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    ticket, messages = ticket_with_messages
    
    # Если summary уже есть в БД, возвращаем его
    if ticket.summary:
        summary = ticket.summary
    else:
        # Иначе генерируем и сохраняем
        rag_service: RAGService = request.app.state.rag
        summary = await rag_service.generate_ticket_summary(messages, ticket_id=conversation_id)
        await crud.update_ticket_summary(session, conversation_id, summary)
    
    return {
        "ticket_id": conversation_id,
        "summary": summary,
        "status": ticket.status,
        "created_at": ticket.created_at.isoformat(),
        "message_count": len(messages)
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
        messages = await crud.list_messages_for_ticket(session, conversation_id, include_system=False)
        history_payload = [_serialize_message(item) for item in messages]

    await websocket.accept()
    await connection_manager.register_chat(conversation_id, websocket)
    try:
        await connection_manager.send_message_history(websocket, conversation_id, history_payload)
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
    """Получить список персонажей для выбора"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    simulator: SimulatorService = request.app.state.simulator
    return {"characters": simulator.CHARACTERS}


@app.post("/api/simulator/start")
async def start_simulator_session(request: Request, character: str = "medium"):
    """Начать новую сессию симулятора"""
    if not auth.is_authenticated_request(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Используем session cookie как user_id
    settings = auth.get_settings()
    user_id = request.cookies.get(settings.cookie_name, "anonymous")
    simulator: SimulatorService = request.app.state.simulator
    
    # Начинаем сессию
    session = simulator.start_session(user_id, character)
    
    # Генерируем первый вопрос
    question = simulator.generate_question(session)
    
    return {
        "session_id": user_id,
        "character": session.character,
        "character_info": simulator.CHARACTERS.get(session.character, {}),
        "questions_total": session.questions_count,
        "current_question": session.current_question + 1,
        "question": question.question,
    }


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
    
    # Сохраняем результат
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
    
    # Если не завершена - генерируем следующий вопрос
    if not is_complete:
        next_question = simulator.generate_question(session)
        response_data["next_question"] = next_question.question
    else:
        # Сессия завершена - отправляем статистику
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
    """Завершить сессию досрочно"""
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






