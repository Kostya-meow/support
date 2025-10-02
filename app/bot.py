from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app import tickets_crud as crud, models
from app.rag_service import RAGResult, RAGService
from app.realtime import ConnectionManager
from app.schemas import TicketRead, MessageRead

logger = logging.getLogger(__name__)

USER_SENDER = "user"
BOT_SENDER = "bot"
OPERATOR_REQUEST_CALLBACK = "request_operator"

REQUEST_OPERATOR_KEYBOARD = InlineKeyboardMarkup(
    inline_keyboard=[[InlineKeyboardButton(text="Позвать оператора", callback_data=OPERATOR_REQUEST_CALLBACK)]]
)


def _extract_title(message: Message = None, user_obj = None) -> str:
    """Извлекает имя пользователя из сообщения или объекта пользователя"""
    user = user_obj if user_obj else (message.from_user if message else None)
    if user:
        if user.full_name:
            return user.full_name
        if user.username:
            return f"@{user.username}"
        if user.first_name:
            return user.first_name
    
    # Fallback к chat_id если нет информации о пользователе
    if message:
        return f"Пользователь {message.chat.id}"
    return "Неизвестный пользователь"


def create_dispatcher(
    session_maker: async_sessionmaker[AsyncSession],
    connection_manager: ConnectionManager,
    rag_service: RAGService,
) -> Dispatcher:
    router = Router()

    async def _serialize_tickets(session: AsyncSession) -> list[dict]:
        tickets = await crud.list_tickets(session, archived=False)
        return [TicketRead.from_orm(item).model_dump(mode="json") for item in tickets]

    async def _broadcast_tickets() -> None:
        async with session_maker() as session:
            tickets_payload = await _serialize_tickets(session)
        await connection_manager.broadcast_conversations(tickets_payload)

    async def _broadcast_message(conversation_id: int, message_schema: MessageRead) -> None:
        await connection_manager.broadcast_message(conversation_id, message_schema.model_dump(mode="json"))

    async def _persist_message(message: Message, text: str, sender: str = USER_SENDER) -> tuple[int | None, bool]:
        """Сохраняет сообщение в существующую заявку (если есть)"""
        chat_id = message.chat.id
        async with session_maker() as session:
            # Ищем ОТКРЫТУЮ заявку для данного чата
            ticket = await crud.get_open_ticket_by_chat_id(session, chat_id)
            if ticket is None:
                # Нет открытой заявки - это обычное общение с ботом
                return None, False
            
            # Есть открытая заявка - добавляем сообщение
            db_message = await crud.add_message(
                session,
                ticket_id=ticket.id,
                sender=sender,
                text=text,
                telegram_message_id=message.message_id if sender == USER_SENDER else None,
            )
            
        await _broadcast_message(ticket.id, MessageRead.from_orm(db_message))
        await _broadcast_tickets()
        return ticket.id, True

    async def _create_ticket_and_add_message(message: Message, text: str) -> int:
        """Создает новую заявку и добавляет первое сообщение"""
        chat_id = message.chat.id
        title = _extract_title(message)
        async with session_maker() as session:
            # Создаем новую заявку
            ticket = await crud.create_ticket(session, chat_id, title)
            
            # Пытаемся получить и добавить историю чата из RAG сервиса
            try:
                print(f"BOT DEBUG: Creating ticket for user {chat_id}, current message: {text}")
                
                # Получаем историю с момента последней заявки или с начала
                chat_history = rag_service.get_chat_history_since_last_ticket(chat_id)
                print(f"BOT DEBUG: Retrieved segmented chat history for user {chat_id}: {len(chat_history)} messages")
                logger.info(f"Retrieved segmented chat history for user {chat_id}: {len(chat_history)} messages")
                
                # Логируем содержимое истории для отладки
                for i, msg in enumerate(chat_history):
                    sender_type = "USER" if msg.is_user else "BOT"
                    print(f"BOT DEBUG: History message {i+1}: [{sender_type}] {msg.message[:50]}...")
                    logger.debug(f"History message {i+1}: [{sender_type}] {msg.message[:50]}...")
                
                # Добавляем все сообщения из релевантной истории чата
                for i, chat_msg in enumerate(chat_history):
                    try:
                        sender = USER_SENDER if chat_msg.is_user else BOT_SENDER
                        await crud.add_message(
                            session,
                            ticket_id=ticket.id,
                            sender=sender,
                            text=chat_msg.message,
                            created_at=chat_msg.timestamp,
                            is_system=False
                        )
                        print(f"BOT DEBUG: Added history message {i+1}/{len(chat_history)}: {sender} - {chat_msg.message[:30]}...")
                        logger.debug(f"Added history message {i+1}/{len(chat_history)}: {sender}")
                    except Exception as e:
                        print(f"BOT DEBUG: Failed to add history message {i+1}: {e}")
                        logger.warning(f"Failed to add history message {i+1}: {e}")
                
                # Отмечаем создание заявки
                rag_service.mark_ticket_created(chat_id)
                
                # НЕ очищаем историю чата - оставляем для будущих заявок
                logger.info(f"Marked ticket creation for user {chat_id}")
                
                # Проверяем, есть ли уже текущее сообщение в истории
                last_user_messages = [msg.message for msg in chat_history if msg.is_user]
                if text in last_user_messages:
                    # Текущее сообщение уже есть в истории, не дублируем
                    # Берем последнее добавленное сообщение как db_message для ответа
                    messages = await crud.list_messages_for_ticket(session, ticket.id)
                    db_message = messages[-1] if messages else None
                    if not db_message:
                        # Если по какой-то причине сообщений нет, добавляем текущее
                        raise Exception("No messages found after adding history")
                else:
                    # Добавляем текущее сообщение, если его нет в истории
                    db_message = await crud.add_message(
                        session,
                        ticket_id=ticket.id,
                        sender=USER_SENDER,
                        text=text,
                        telegram_message_id=message.message_id,
                    )
                    
            except Exception as e:
                # Если что-то пошло не так с историей, просто добавляем текущее сообщение
                logger.warning(f"Failed to process chat history for user {chat_id}: {e}")
                db_message = await crud.add_message(
                    session,
                    ticket_id=ticket.id,
                    sender=USER_SENDER,
                    text=text,
                    telegram_message_id=message.message_id,
                )
            
        await _broadcast_message(ticket.id, MessageRead.from_orm(db_message))
        await _broadcast_tickets()
        return ticket.id

    async def _answer_with_rag_only(message: Message, user_text: str) -> None:
        """Отвечает пользователю через RAG без создания заявки"""
        try:
            # Отправляем typing action
            try:
                await message.bot.send_chat_action(message.chat.id, 'typing')
            except Exception:
                pass
            # Используем chat_id как temporary conversation_id для RAG
            conversation_id = message.chat.id
            rag_result: RAGResult = await asyncio.to_thread(
                rag_service.generate_reply,
                conversation_id,
                user_text,
            )
        except Exception as exc:
            logger.exception("RAG generation failed: %s", exc)
            fallback = "Не смогла обработать запрос. Попробуйте ещё раз или позовите оператора."
            await message.answer(fallback, reply_markup=REQUEST_OPERATOR_KEYBOARD)
            return

        if rag_result.operator_requested:
            # RAG решил, что нужен оператор - создаем заявку
            ticket_id = await _create_ticket_and_add_message(message, user_text)
            if rag_result.final_answer:
                await message.answer(rag_result.final_answer)
                await _send_bot_message(ticket_id, rag_result.final_answer)
            return

        # Обычный ответ от бота
        answer_text = rag_result.final_answer or "Я пока не нашла ответ. Попробуйте уточнить вопрос."
        await message.answer(answer_text, reply_markup=REQUEST_OPERATOR_KEYBOARD)

    async def _send_bot_message(ticket_id: int, text: str, is_system: bool = False) -> None:
        async with session_maker() as session:
            db_message = await crud.add_message(session, ticket_id, BOT_SENDER, text, is_system=is_system)
            tickets_payload = await _serialize_tickets(session)
        await _broadcast_message(ticket_id, MessageRead.from_orm(db_message))
        await connection_manager.broadcast_conversations(tickets_payload)

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        # При /start всегда отвечаем ботом, заявку не создаем
        greeting = (
            "Здравствуйте! Опишите проблему — постараюсь помочь. Если ответ не подойдёт, можно позвать оператора."
        )
        await message.answer(greeting, reply_markup=REQUEST_OPERATOR_KEYBOARD)

    @router.message(F.voice)
    async def on_voice(message: Message) -> None:
        """Обработчик голосовых сообщений"""
        try:
            # Получаем файл голосового сообщения
            voice = message.voice
            if not voice:
                await message.answer("Не удалось получить голосовое сообщение.")
                return
            
            # Уведомляем пользователя о начале обработки
            processing_msg = await message.answer("🎤 Обрабатываю голосовое сообщение...")
            
            # Получаем Bot из параметров dispatcher
            bot = message.bot
            
            # Скачиваем файл во временную папку
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
                await bot.download(voice.file_id, temp_file.name)
                temp_file_path = temp_file.name
            
            try:
                # Преобразуем голос в текст
                transcribed_text = await rag_service.speech_to_text.transcribe_audio(temp_file_path)
                
                if not transcribed_text:
                    await processing_msg.edit_text("❌ Не удалось распознать речь. Попробуйте еще раз или напишите текстом.")
                    return
                
                # Удаляем сообщение "Обрабатываю..."
                await processing_msg.delete()

                # Сохраняем расшифровку только для оператора
                ticket_id, has_ticket = await _persist_message(message, transcribed_text)

                if has_ticket and ticket_id:
                    # В чат оператора отправляем только текст
                    await _send_bot_message(ticket_id, transcribed_text)
                else:
                    # Для пользователя просто ответ бота
                    await _answer_with_rag_only(message, transcribed_text)
                
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error processing voice message: {e}")
            await message.answer("❌ Произошла ошибка при обработке голосового сообщения. Попробуйте написать текстом.")

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        user_text = message.text or ""
        
        # Проверяем, есть ли открытая заявка
        ticket_id, has_ticket = await _persist_message(message, user_text)
        
        if has_ticket and ticket_id:
            # Есть открытая заявка - пользователь общается с оператором
            # Ничего не отвечаем от бота
            return
        else:
            # Нет открытой заявки - обычное общение с ботом
            await _answer_with_rag_only(message, user_text)

    @router.message(F.caption)
    async def on_caption(message: Message) -> None:
        caption_text = message.caption or ""
        
        # Проверяем, есть ли открытая заявка
        ticket_id, has_ticket = await _persist_message(message, caption_text)
        
        warning = "Пока могу обрабатывать только текстовые сообщения."
        await message.answer(warning, reply_markup=REQUEST_OPERATOR_KEYBOARD)
        
        if has_ticket and ticket_id:
            await _send_bot_message(ticket_id, warning)
        elif caption_text:
            # Нет заявки, но есть текст - отвечаем через RAG
            await _answer_with_rag_only(message, caption_text)

    @router.message()
    async def on_other(message: Message) -> None:
        # Проверяем, есть ли открытая заявка
        ticket_id, has_ticket = await _persist_message(message, "[unsupported message type]")
        
        warning = "Пока поддерживаются только текстовые сообщения. Пожалуйста, напишите ваш вопрос."
        await message.answer(warning, reply_markup=REQUEST_OPERATOR_KEYBOARD)
        
        if has_ticket and ticket_id:
            await _send_bot_message(ticket_id, warning)

    @router.callback_query(lambda query: query.data == OPERATOR_REQUEST_CALLBACK)
    async def on_request_operator(callback_query: CallbackQuery) -> None:
        chat = callback_query.message.chat if callback_query.message else None
        if chat is None:
            await callback_query.answer()
            return
        
        async with session_maker() as session:
            # Ищем открытую заявку
            ticket = await crud.get_open_ticket_by_chat_id(session, chat.id)
            if ticket is None:
                print(f"BOT DEBUG: Creating new ticket via operator button for chat {chat.id}")
                
                # Получаем историю чата для создания заявки с контекстом
                chat_history = rag_service.get_chat_history_since_last_ticket(chat.id)
                print(f"BOT DEBUG: Retrieved chat history: {len(chat_history)} messages")
                
                # Находим последнее сообщение пользователя для заголовка заявки
                last_user_message = "Запрос помощи"
                for msg in reversed(chat_history):
                    if msg.is_user and not msg.message.startswith('/'):
                        last_user_message = msg.message
                        break
                
                print(f"BOT DEBUG: Using last user message for ticket: {last_user_message[:50]}...")
                
                # Создаем заявку с историей
                title = _extract_title(user_obj=callback_query.from_user) if callback_query.from_user else f"Заявка от {chat.id}"
                ticket = await crud.create_ticket(session, chat.id, title)
                
                # Добавляем всю историю чата в заявку
                for i, chat_msg in enumerate(chat_history):
                    try:
                        sender = USER_SENDER if chat_msg.is_user else BOT_SENDER
                        await crud.add_message(
                            session,
                            ticket_id=ticket.id,
                            sender=sender,
                            text=chat_msg.message,
                            created_at=chat_msg.timestamp,
                            is_system=False
                        )
                        print(f"BOT DEBUG: Added history message {i+1}: [{sender}] {chat_msg.message[:30]}...")
                    except Exception as e:
                        print(f"BOT DEBUG: Failed to add history message {i+1}: {e}")
                
                # Отмечаем создание заявки в RAG сервисе
                rag_service.mark_ticket_created(chat.id)
                print(f"BOT DEBUG: Added {len(chat_history)} messages to ticket {ticket.id}")
            
            # Переводим в статус "открыта" если была в работе
            if ticket.status != models.TicketStatus.OPEN:
                await crud.update_ticket_status(session, ticket.id, models.TicketStatus.OPEN)
            
            tickets_payload = await _serialize_tickets(session)
            
        rag_service.reset_history(ticket.id)
        await callback_query.answer("Оператор скоро подключится")
        await connection_manager.broadcast_conversations(tickets_payload)
        
        notice = "Мы уведомили оператора, ожидайте ответа."
        await callback_query.message.answer(notice)
        await _send_bot_message(ticket.id, notice, is_system=True)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher


async def start_bot(bot: Bot, dispatcher: Dispatcher) -> None:
    try:
        await dispatcher.start_polling(bot)
    except asyncio.CancelledError:
        raise



