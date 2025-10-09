from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections import defaultdict

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

# Словарь блокировок для каждого пользователя (предотвращение спама)
user_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

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
        result = []
        for ticket in tickets:
            ticket_data = TicketRead.from_orm(ticket).model_dump(mode="json")
            # Подсчитываем непрочитанные сообщения от user и bot
            unread_count = sum(
                1 for msg in ticket.messages 
                if msg.sender in ['user', 'bot'] and not msg.is_read
            )
            # Ограничиваем максимум 99
            ticket_data['unread_count'] = min(unread_count, 99)
            result.append(ticket_data)
        return result

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
            
            # Проверяем, есть ли активные подключения к этому чату
            # Если есть - сразу помечаем сообщение как прочитанное
            should_mark_as_read = connection_manager.has_active_chat_connections(ticket.id)
            logger.info(f"📨 Message from {sender} to ticket #{ticket.id}: has_active={should_mark_as_read}")
            
            # Есть открытая заявка - добавляем сообщение
            db_message = await crud.add_message(
                session,
                ticket_id=ticket.id,
                sender=sender,
                text=text,
                telegram_message_id=message.message_id if sender == USER_SENDER else None,
                is_read=should_mark_as_read,  # Сразу помечаем как прочитанное, если чат открыт
            )
            logger.info(f"✅ Message #{db_message.id} created with is_read={db_message.is_read}")
            
            # add_message уже делает commit, просто загружаем актуальные данные
            await session.refresh(db_message)  # Обновляем объект
            # Загружаем заявку с сообщениями для правильного подсчета unread
            await session.refresh(ticket, ['messages'])
            tickets_payload = await _serialize_tickets(session)
            
        await _broadcast_message(ticket.id, MessageRead.from_orm(db_message))
        await connection_manager.broadcast_conversations(tickets_payload)
        return ticket.id, True

    async def _create_ticket_and_add_message(message: Message, text: str) -> int:
        """Создает новую заявку и добавляет первое сообщение"""
        chat_id = message.chat.id
        title = _extract_title(message)
        async with session_maker() as session:
            # Создаем новую заявку
            ticket = await crud.create_ticket(session, chat_id, title)
            
            # Проверяем, есть ли активные подключения к этому чату (маловероятно для новой заявки, но проверим)
            should_mark_as_read = connection_manager.has_active_chat_connections(ticket.id)
            
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
                        # Для новых сообщений не передавать created_at, чтобы использовалось текущее время
                        await crud.add_message(
                            session,
                            ticket_id=ticket.id,
                            sender=sender,
                            text=chat_msg.message,
                            is_system=False,
                            is_read=should_mark_as_read
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
                        is_read=should_mark_as_read
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
                    is_read=should_mark_as_read
                )
            
            # Генерируем summary сразу после создания заявки
            try:
                messages = await crud.list_messages_for_ticket(session, ticket.id)
                summary = await rag_service.generate_ticket_summary(messages, ticket_id=ticket.id)
                # Сохраняем summary в БД
                await crud.update_ticket_summary(session, ticket.id, summary)
                logger.info(f"Generated and saved summary for ticket {ticket.id}: {summary[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to generate summary for ticket {ticket.id}: {e}")
            
        await _broadcast_message(ticket.id, MessageRead.from_orm(db_message))
        await _broadcast_tickets()
        return ticket.id

    async def _get_average_response_time() -> str:
        """Вычисляет среднее время ответа оператора"""
        try:
            async with session_maker() as session:
                from sqlalchemy import select, func
                from datetime import datetime, timedelta
                
                # Получаем закрытые заявки за последние 30 дней
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                stmt = select(models.Ticket).where(
                    models.Ticket.status == models.TicketStatus.CLOSED,
                    models.Ticket.created_at >= thirty_days_ago
                )
                result = await session.execute(stmt)
                tickets = result.scalars().all()
                
                if not tickets:
                    return "обычно быстро"
                
                # Вычисляем среднее время между созданием и первым ответом оператора
                response_times = []
                for ticket in tickets:
                    messages = await crud.list_messages_for_ticket(session, ticket.id)
                    # Находим первое сообщение оператора
                    operator_message = next((m for m in messages if m.sender == "operator"), None)
                    if operator_message and ticket.created_at:
                        delta = operator_message.created_at - ticket.created_at
                        response_times.append(delta.total_seconds() / 60)  # в минутах
                
                if not response_times:
                    return "обычно быстро"
                
                avg_minutes = sum(response_times) / len(response_times)
                
                if avg_minutes < 1:
                    return "менее минуты"
                elif avg_minutes < 5:
                    return f"{int(avg_minutes)} мин"
                elif avg_minutes < 60:
                    return f"{int(avg_minutes)} минут"
                else:
                    hours = int(avg_minutes / 60)
                    return f"около {hours} ч"
                    
        except Exception as e:
            logger.warning(f"Failed to calculate average response time: {e}")
            return "обычно быстро"

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
            # RAG решил, что нужен оператор - показываем предложение
            avg_response_time = await _get_average_response_time()
            
            # RAG уже вернул текст типа "Могу подключить оператора", не дублируем!
            combined_text = (
                f"<b>Бот:</b>\n"
                f"{rag_result.final_answer}\n\n"
                f"⏱ Среднее время ответа: <b>{avg_response_time}</b>\n\n"
                f"Подключить?"
            )
            
            # Создаем клавиатуру с кнопкой подтверждения
            confirm_keyboard = InlineKeyboardMarkup(
                inline_keyboard=[[
                    InlineKeyboardButton(text="✅ Да, подключить оператора", callback_data=OPERATOR_REQUEST_CALLBACK)
                ]]
            )
            
            await message.answer(combined_text, reply_markup=confirm_keyboard, parse_mode='HTML')
            return

        # Обычный ответ от бота
        answer_text = rag_result.final_answer or "Я пока не нашла ответ. Попробуйте уточнить вопрос."
        formatted_answer = f"<b>Бот:</b>\n{answer_text}"
        
        # Показываем кнопку оператора ТОЛЬКО если уверенность низкая (confidence_score > 0.6)
        # confidence_score - это evaluation score, чем выше, тем хуже качество ответа
        if rag_result.confidence_score > 0.6:
            # Низкая уверенность - предлагаем оператора с тем же единым форматом
            avg_response_time = await _get_average_response_time()
            
            combined_text = (
                f"<b>Бот:</b>\n{answer_text}\n\n"
                f"Могу подключить оператора.\n"
                f"⏱ Среднее время ответа: <b>{avg_response_time}</b>\n\n"
                f"Подключить?"
            )
            
            confirm_keyboard = InlineKeyboardMarkup(
                inline_keyboard=[[
                    InlineKeyboardButton(text="✅ Да, подключить оператора", callback_data=OPERATOR_REQUEST_CALLBACK)
                ]]
            )
            
            await message.answer(combined_text, reply_markup=confirm_keyboard, parse_mode='HTML')
        else:
            # Уверенный ответ - показываем ответ и (опционально) темы-быстрые кнопки
            try:
                await message.answer(formatted_answer, parse_mode='HTML')
            except Exception:
                logger.exception("Failed to send confident answer to chat %s", message.chat.id)

            # Suggest topics (short quick-follow buttons)
            try:
                topics = await asyncio.to_thread(rag_service.suggest_topics, conversation_id, user_text, answer_text)
                if topics:
                    buttons = []
                    for t in topics:
                        cb = f"topic::{t}"
                        # If callback_data might be long, we could map to a short id — for now keep it simple
                        buttons.append([InlineKeyboardButton(text=t, callback_data=cb)])
                    topic_kb = InlineKeyboardMarkup(inline_keyboard=buttons)
                    await message.answer("Хотите узнать подробнее по этим темам:", reply_markup=topic_kb)
            except Exception:
                logger.exception("Failed to generate or send topic suggestions for chat %s", message.chat.id)

    @router.callback_query(F.data.startswith('topic::'))
    async def on_topic_callback(query: CallbackQuery) -> None:
        # quick handler: emulate user asking the topic question
        data = (query.data or '')
        topic = data.split('::', 1)[1] if '::' in data else data
        try:
            await query.answer()  # remove loading
        except Exception:
            pass
        # Send typing and call RAG as if user asked topic
        try:
            chat_id = query.message.chat.id if query.message else query.from_user.id
            try:
                await query.bot.send_chat_action(chat_id, 'typing')
            except Exception:
                pass
            # Generate reply synchronously in thread
            rag_result: RAGResult = await asyncio.to_thread(rag_service.generate_reply, chat_id, topic)
            # Send back to chat
            text = rag_result.final_answer or 'Нет ответа.'
            try:
                if query.message:
                    await query.message.answer(text)
                else:
                    await query.bot.send_message(query.from_user.id, text)
            except Exception:
                logger.exception('Failed to send topic reply')
        except Exception:
            logger.exception('Error handling topic callback')

    async def _send_bot_message(ticket_id: int, text: str, is_system: bool = False) -> None:
        # Проверяем, есть ли активные подключения к этому чату
        should_mark_as_read = connection_manager.has_active_chat_connections(ticket_id)
        async with session_maker() as session:
            db_message = await crud.add_message(
                session, ticket_id, BOT_SENDER, text, 
                is_system=is_system, 
                is_read=should_mark_as_read
            )
            # add_message уже делает commit, просто обновляем объект
            await session.refresh(db_message)
            tickets_payload = await _serialize_tickets(session)
        await _broadcast_message(ticket_id, MessageRead.from_orm(db_message))
        await connection_manager.broadcast_conversations(tickets_payload)

    @router.message(CommandStart())
    async def on_start(message: Message) -> None:
        # При /start всегда отвечаем ботом, заявку не создаем
        user_name = message.from_user.first_name if message.from_user else "пользователь"
        greeting = (
            f"👋 Здравствуйте, {user_name}!\n\n"
            "Я бот технической поддержки. Я могу помочь вам с:\n"
            "• Ответами на часто задаваемые вопросы\n"
            "• Решением технических проблем\n"
            "• Консультацией по функциям системы\n\n"
            "📚 <b>Популярные вопросы и ответы:</b>\n"
            "Посмотрите наш FAQ: http://127.0.0.1:8000/faq\n\n"
            "💬 Просто напишите ваш вопрос, и я постараюсь помочь!\n"
            "Если мой ответ не подойдет, я смогу подключить оператора."
        )
        formatted_greeting = f"<b>Бот:</b>\n{greeting}"
        await message.answer(formatted_greeting, parse_mode='HTML')

    @router.message(F.voice)
    async def on_voice(message: Message) -> None:
        """Обработчик голосовых сообщений"""
        chat_id = message.chat.id
        lock = user_locks[chat_id]
        
        # Проверяем блокировку
        if lock.locked():
            logger.info(f"User {chat_id} is spamming voice messages, ignoring")
            await message.answer("⏳ Пожалуйста, дождитесь ответа на предыдущее сообщение")
            return
        
        async with lock:
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

                    # Если есть заявка, сообщение уже сохранено как от пользователя — ничего не отправляем от бота
                    # Если нет заявки — просто ответить пользователю
                    if not has_ticket or not ticket_id:
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
        chat_id = message.chat.id
        
        # Получаем блокировку для этого пользователя
        lock = user_locks[chat_id]
        
        # Проверяем, не обрабатывается ли уже другое сообщение
        if lock.locked():
            logger.info(f"User {chat_id} is spamming, ignoring message")
            await message.answer("⏳ Пожалуйста, дождитесь ответа на предыдущее сообщение")
            return
        
        # Захватываем блокировку на время обработки
        async with lock:
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
                
                # Проверяем, есть ли активные подключения
                should_mark_as_read = connection_manager.has_active_chat_connections(ticket.id)
                
                # Добавляем всю историю чата в заявку
                for i, chat_msg in enumerate(chat_history):
                    try:
                        sender = USER_SENDER if chat_msg.is_user else BOT_SENDER
                        await crud.add_message(
                            session,
                            ticket_id=ticket.id,
                            sender=sender,
                            text=chat_msg.message,
                            is_system=False,
                            is_read=should_mark_as_read
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
            
            # Генерируем summary для новой заявки
            try:
                messages = await crud.list_messages_for_ticket(session, ticket.id)
                if messages:
                    summary = await rag_service.generate_ticket_summary(messages, ticket_id=ticket.id)
                    await crud.update_ticket_summary(session, ticket.id, summary)
                    logger.info(f"Generated summary for ticket {ticket.id}")
            except Exception as e:
                logger.warning(f"Failed to generate summary: {e}")
            
            tickets_payload = await _serialize_tickets(session)
            
        rag_service.reset_history(ticket.id)
        await callback_query.answer("✅ Заявка создана")
        await connection_manager.broadcast_conversations(tickets_payload)
        
        notice = "✅ Заявка создана. Ожидайте ответа оператора."
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



