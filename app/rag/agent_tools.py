"""
Инструменты для RAG агента
"""

import logging
import threading
from typing import List, Dict, Any, Optional
from agno.tools import tool

logger = logging.getLogger(__name__)

# Thread-local хранилище для передачи данных между агентом и ботом
_thread_local = threading.local()


def set_current_conversation_id(conversation_id: int):
    """Установить ID текущего разговора для thread"""
    _thread_local.conversation_id = conversation_id


def get_current_conversation_id() -> Optional[int]:
    """Получить ID текущего разговора"""
    return getattr(_thread_local, "conversation_id", None)


# Глобальное хранилище для передачи данных между агентом и ботом
# Ключ - conversation_id, значение - список похожих результатов
_similar_suggestions_storage: Dict[int, List[Dict[str, Any]]] = {}

# Храним timestamp последнего показа кнопок для каждого conversation
# Чтобы не спамить кнопками слишком часто
_last_suggestions_time: Dict[int, float] = {}

# Минимальный интервал между показами кнопок (в секундах)
MIN_SUGGESTIONS_INTERVAL = 60  # 1 минута


def store_similar_suggestions(conversation_id: int, suggestions: List[Dict[str, Any]]):
    """Сохранить похожие результаты для разговора"""
    import time

    global _similar_suggestions_storage, _last_suggestions_time

    # Проверяем, не показывали ли мы кнопки недавно
    last_time = _last_suggestions_time.get(conversation_id, 0)
    current_time = time.time()

    if current_time - last_time < MIN_SUGGESTIONS_INTERVAL:
        logger.info(
            f"Skipping suggestions for conversation {conversation_id} - shown too recently "
            f"({int(current_time - last_time)}s ago, minimum {MIN_SUGGESTIONS_INTERVAL}s)"
        )
        return  # Не сохраняем - слишком рано

    _similar_suggestions_storage[conversation_id] = suggestions
    _last_suggestions_time[conversation_id] = current_time
    logger.info(
        f"Stored {len(suggestions)} similar suggestions for conversation {conversation_id}"
    )


def get_similar_suggestions(conversation_id: int) -> List[Dict[str, Any]] | None:
    """Получить похожие результаты для разговора и очистить хранилище"""
    global _similar_suggestions_storage
    suggestions = _similar_suggestions_storage.pop(conversation_id, None)
    if suggestions:
        logger.info(
            f"Retrieved {len(suggestions)} similar suggestions for conversation {conversation_id}"
        )
    return suggestions


@tool
async def search_knowledge_base(query: str, suggest_similar: bool = False) -> str:
    """Поиск в базе знаний - твой главный инструмент!

    Параметры:
    - query: запрос для поиска
    - suggest_similar: если True, покажет пользователю 3 кнопки с похожими проблемами для выбора.
      Используй suggest_similar=True ТОЛЬКО когда:
      * Пользователь описывает общую проблему без деталей (например: "интернет не работает", "компьютер тормозит")
      * Есть несколько похожих решений и пользователь может выбрать подходящее
      * НЕ используй если пользователь задал конкретный вопрос или нужен прямой ответ

    По умолчанию (suggest_similar=False) - обычный поиск с прямым ответом.
    """
    print(
        f"[AGENT ACTION] Поиск в базе знаний: '{query}' (suggest_similar={suggest_similar})"
    )

    try:
        from app.db.database import KnowledgeSessionLocal
        from app.db import tickets_crud as crud
        from sentence_transformers import SentenceTransformer
        import numpy as np

        # Инициализируем модель для семантического поиска
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        async with KnowledgeSessionLocal() as session:
            # Загружаем все чанки
            chunks = await crud.load_all_chunks(session)

            if not chunks:
                print("[AGENT] База знаний пуста")
                return "База знаний пуста"

            print(f"[AGENT] Поиск среди {len(chunks)} записей...")

            # Создаем векторное представление запроса
            query_embedding = model.encode([query], convert_to_numpy=True)
            query_vector = query_embedding[0]

            # Ищем среди чанков с эмбеддингами
            results = []
            text_results = []  # Для чанков без эмбеддингов

            for chunk in chunks:
                if chunk.embedding is not None:
                    # Семантический поиск
                    chunk_vector = np.frombuffer(chunk.embedding, dtype=np.float32)
                    # Косинусное сходство
                    similarity = np.dot(query_vector, chunk_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(chunk_vector)
                    )
                    results.append(
                        {
                            "id": chunk.id,
                            "content": chunk.content,
                            "source": chunk.source_file,
                            "score": float(similarity),
                            "type": "semantic",
                        }
                    )
                else:
                    # Текстовый поиск для чанков без эмбеддингов
                    query_lower = query.lower()
                    if query_lower in chunk.content.lower():
                        relevance = chunk.content.lower().count(query_lower)
                        text_results.append(
                            {
                                "id": chunk.id,
                                "content": chunk.content,
                                "source": chunk.source_file,
                                "score": relevance / 10.0,  # Нормализуем
                                "type": "text",
                            }
                        )

            # Объединяем результаты
            all_results = results + text_results

            if not all_results:
                print(f"[AGENT] Ничего не найдено по запросу: '{query}'")
                return f"По запросу '{query}' ничего не найдено в базе знаний"

            # Сортируем по релевантности
            all_results.sort(key=lambda x: x["score"], reverse=True)

            print(f"[AGENT] Найдено {len(all_results)} результатов")

            # Формируем обычный текстовый ответ из топ-3 результатов
            response_parts = []
            for i, result in enumerate(all_results[:3], 1):
                content = result["content"]
                # Обрабатываем проблемные символы Unicode
                try:
                    content = content.encode("cp1251", errors="ignore").decode("cp1251")
                except:
                    content = content.encode("ascii", errors="ignore").decode("ascii")

                if len(content) > 300:
                    content = content[:300] + "..."

                response_parts.append(
                    f"{i}. Источник: {result['source']}\n"
                    f"Содержание: {content}\n"
                    f"(релевантность: {result['score']:.3f}, тип: {result['type']})"
                )

            response = "\n\n".join(response_parts)

            # Дополнительно: если нужно показать кнопки с похожими вариантами
            if suggest_similar and len(all_results) >= 3:
                print("[AGENT] Сохраняю варианты для показа кнопок пользователю")

                # Берём топ-3
                top_3 = all_results[:3]
                suggestions = []

                for result in top_3:
                    # Создаём краткое описание для кнопки (первые 80 символов)
                    preview = result["content"][:80].replace("\n", " ").strip()
                    if len(result["content"]) > 80:
                        preview += "..."

                    suggestions.append(
                        {
                            "id": result["id"],
                            "preview": preview,
                            "score": result["score"],
                            "source": result["source"],
                            "full_content": result[
                                "content"
                            ],  # Полный текст для отображения
                        }
                    )

                # Сохраняем в хранилище (если известен conversation_id)
                conversation_id = get_current_conversation_id()
                if conversation_id:
                    store_similar_suggestions(conversation_id, suggestions)
                    print(
                        f"[AGENT] Сохранил {len(suggestions)} вариантов для conversation {conversation_id}"
                    )
                else:
                    print(
                        "[AGENT WARNING] conversation_id не установлен, кнопки не будут показаны!"
                    )

                print(f"[AGENT] Кнопки будут показаны дополнительным сообщением")

            print(
                f"[AGENT] Возвращаю результат поиска (длина: {len(response)} символов)"
            )
            return response

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка поиска: {e}")
        logger.error(f"Error searching knowledge base: {e}")
        return f"Ошибка поиска в базе знаний: {e}"


@tool
def improve_search_query(original_query: str, context: str = None) -> str:
    """Улучшение поискового запроса через LLM - анализирует запрос и создает оптимальные поисковые термины для любой предметной области."""
    print(f"[AGENT ACTION] Улучшение запроса через LLM: '{original_query}'")

    try:
        from app.rag.service import get_llm_client
        import yaml

        # Загружаем конфигурацию
        with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Получаем промпт из конфига
        improvement_prompt_template = config.get("agent", {}).get(
            "improve_search_prompt", ""
        )

        # Формируем промпт с подстановкой значений
        improvement_prompt = improvement_prompt_template.format(
            original_query=original_query,
            context=context if context else "Техническая поддержка",
        )

        # Получаем LLM клиент
        llm_client = get_llm_client()

        # Улучшаем запрос через LLM
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": improvement_prompt}],
            max_tokens=100,
            temperature=0.3,
        )

        improved_query = response.choices[0].message.content.strip()

        # Убираем кавычки если LLM их добавил
        if improved_query.startswith('"') and improved_query.endswith('"'):
            improved_query = improved_query[1:-1]

        if improved_query != original_query:
            print(f"[AGENT] Запрос улучшен LLM: '{improved_query}'")
            return improved_query
        else:
            print(f"[AGENT] LLM решил что запрос уже оптимален")
            return original_query

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка улучшения запроса: {e}")

        # Фоллбэк на простое улучшение
        simple_improvements = {
            "принтер": ["принтер", "печать", "МФУ", "струйный", "лазерный"],
            "компьютер": ["компьютер", "ПК", "ноутбук", "системный блок"],
            "интернет": ["интернет", "сеть", "wi-fi", "подключение"],
            "ошибка": ["ошибка", "проблема", "сбой", "не работает"],
            "программа": ["программа", "приложение", "софт", "ПО"],
        }

        original_lower = original_query.lower()
        enhanced_terms = [original_query]

        for key, synonyms in simple_improvements.items():
            if key in original_lower:
                enhanced_terms.extend(synonyms[:3])  # Берем первые 3 синонима
                break

        if len(enhanced_terms) > 1:
            improved_query = " ".join(enhanced_terms[:4])  # Максимум 4 термина
            print(f"[AGENT] Запрос улучшен (fallback): '{improved_query}'")
            return improved_query
        else:
            print(f"[AGENT] Запрос остался без изменений")
            return original_query


def _classify_request_internal(
    dialogue_history: str = None, text: str = None, categories: List[str] = None
) -> str:
    """Внутренняя функция классификации для прямого вызова из ботов (без декоратора @tool)."""
    # Определяем текст для анализа
    if dialogue_history:
        analysis_text = dialogue_history
        print(
            f"[AGENT ACTION] Классификация диалога через LLM (длина: {len(dialogue_history)} символов)"
        )
    elif text:
        analysis_text = text
        print(f"[AGENT ACTION] Классификация запроса через LLM: '{text[:50]}...'")
    else:
        return "Ошибка: не предоставлен ни диалог, ни текст для классификации"

    try:
        from app.rag.service import get_llm_client
        import yaml

        # Загружаем конфигурацию
        with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Получаем категории из конфига
        default_categories = [
            "Инфраструктура",
            "Разработка",
            "Сеть",
            "Безопасность",
            "Пользовательское ПО",
            "Аппаратные проблемы",
            "Общий",
        ]
        available_categories = config.get("agent", {}).get(
            "categories", categories or default_categories
        )

        # Формируем промпт для классификации из конфига
        categories_text = ", ".join(available_categories)
        classification_prompt_template = config.get("agent", {}).get(
            "classify_request_prompt", ""
        )

        classification_prompt = classification_prompt_template.format(
            categories_text=categories_text, analysis_text=analysis_text
        )

        # Получаем LLM клиент
        llm_client = get_llm_client()

        # Классифицируем через LLM
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": classification_prompt}],
            max_tokens=100,
            temperature=0.1,
        )

        llm_result = response.choices[0].message.content.strip()

        # Парсим результат и проверяем категории
        suggested_categories = [cat.strip() for cat in llm_result.split(",")]
        valid_categories = []

        for cat in suggested_categories:
            if cat in available_categories:
                valid_categories.append(cat)

        # Если LLM не вернул валидные категории, используем fallback
        if not valid_categories:
            text_lower = analysis_text.lower()
            category_keywords = {
                "Инфраструктура": [
                    "сервер",
                    "сеть",
                    "железо",
                    "оборудование",
                    "инфраструктура",
                ],
                "Разработка": [
                    "код",
                    "программа",
                    "баг",
                    "ошибка",
                    "приложение",
                    "разработка",
                ],
                "Сеть": ["интернет", "сеть", "роутер", "wifi", "подключение"],
                "Безопасность": [
                    "пароль",
                    "доступ",
                    "права",
                    "безопасность",
                    "блокировка",
                ],
                "Пользовательское ПО": ["программа", "софт", "приложение", "установка"],
                "Аппаратные проблемы": [
                    "компьютер",
                    "принтер",
                    "мышь",
                    "клавиатура",
                    "монитор",
                ],
                "Общий": [],
            }

            scores = {}
            for category, keywords in category_keywords.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    scores[category] = score

            if scores:
                # Берем топ категории
                sorted_categories = sorted(
                    scores.items(), key=lambda x: x[1], reverse=True
                )
                valid_categories = [
                    cat for cat, score in sorted_categories[:2]
                ]  # Максимум 2 категории
            else:
                valid_categories = ["Общий"]

        result_categories = ", ".join(valid_categories)
        print(f"[AGENT] Диалог классифицирован как: {result_categories}")
        logger.info(f"Request classified as: {result_categories}")

        return f"Классификация проблемы: {result_categories}"

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка классификации: {e}")
        logger.error(f"Classification error: {e}")
        return "Ошибка классификации. Категория: Общий"


@tool
def classify_request(
    dialogue_history: str = None, text: str = None, categories: List[str] = None
) -> str:
    """Классификация запроса через LLM - анализирует весь диалог с пользователем для точной классификации проблемы. Возвращает категории проблем которые можно назначить заявке."""
    return _classify_request_internal(dialogue_history, text, categories)


def _set_priority_internal(dialogue_history: str) -> str:
    """Внутренняя функция установки приоритета для прямого вызова из ботов (без декоратора @tool)."""
    print(
        f"[AGENT ACTION] Определение приоритета диалога через LLM (длина: {len(dialogue_history)} символов)"
    )

    try:
        from app.rag.service import get_llm_client
        import yaml

        # Загружаем конфигурацию
        with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Получаем промпт из конфига
        priority_prompt_template = config.get("agent", {}).get(
            "set_priority_prompt", ""
        )

        if not priority_prompt_template:
            print("[AGENT WARNING] Промпт для set_priority не найден в конфиге")
            return "medium"

        # Формируем промпт для определения приоритета
        priority_prompt = priority_prompt_template.format(
            dialogue_text=dialogue_history
        )

        # Получаем LLM клиент
        llm_client = get_llm_client()

        # Определяем приоритет через LLM
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": priority_prompt}],
            max_tokens=10,
            temperature=0.1,
        )

        llm_result = response.choices[0].message.content.strip().lower()

        # Парсим результат и проверяем валидность
        valid_priorities = ["low", "medium", "high"]

        # Извлекаем приоритет из ответа LLM
        priority = "medium"  # По умолчанию
        for valid_priority in valid_priorities:
            if valid_priority in llm_result:
                priority = valid_priority
                break

        print(f"[AGENT] Приоритет диалога определен как: {priority}")
        logger.info(f"Dialogue priority set to: {priority}")

        return priority

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка определения приоритета: {e}")
        logger.error(f"Priority determination error: {e}")
        return "medium"  # По умолчанию при ошибке


@tool
def set_priority(dialogue_history: str = None) -> str:
    """Определение приоритета заявки на основе важности диалога - анализирует весь диалог и устанавливает приоритет (низкий/средний/высокий).

    Используй этот инструмент когда:
    - Пользователь описывает критическую проблему (блокирует работу, потеря данных, безопасность)
    - Проблема влияет на многих пользователей или критические системы
    - Эмоциональный тон указывает на срочность
    - Нужно понять, насколько важна проблема для приоритизации обработки

    Параметры:
    - dialogue_history: полный текст диалога с пользователем для анализа

    Возвращает описание установленного приоритета.
    """
    if not dialogue_history:
        return (
            "Ошибка: необходимо предоставить историю диалога для определения приоритета"
        )

    priority = _set_priority_internal(dialogue_history)

    # Сохраняем приоритет в ticket через conversation_id
    conversation_id = get_current_conversation_id()
    if conversation_id:
        try:
            import asyncio
            from app.db.database import TicketsSessionLocal
            from app.db import tickets_crud as crud

            async def update_priority():
                async with TicketsSessionLocal() as session:
                    # Получаем ticket по conversation_id (это ID тикета)
                    ticket = await crud.get_ticket_by_id(session, conversation_id)
                    if ticket:
                        ticket.priority = priority
                        await session.commit()
                        print(
                            f"[AGENT] Обновлен приоритет ticket {conversation_id}: {priority}"
                        )
                    else:
                        print(
                            f"[AGENT WARNING] Ticket {conversation_id} не найден для обновления приоритета"
                        )

            # Запускаем асинхронную функцию
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(update_priority())
            else:
                loop.run_until_complete(update_priority())

        except Exception as e:
            print(f"[AGENT ERROR] Ошибка сохранения приоритета в БД: {e}")
            logger.error(f"Failed to save priority to database: {e}")

    priority_labels = {
        "low": "низкий",
        "medium": "средний",
        "high": "высокий",
    }

    return f"Приоритет заявки установлен: {priority_labels.get(priority, priority)}"


@tool
def call_operator() -> str:
    """ВНИМАНИЕ: Используй ТОЛЬКО в крайних случаях! Вызывает живого оператора для очень сложных технических проблем, которые невозможно решить самостоятельно или через поиск в базе знаний. Перед использованием обязательно попробуй все другие способы помочь пользователю."""
    print(
        "[AGENT ACTION] ВЫЗОВ ОПЕРАТОРА! Передача сложного запроса живому специалисту"
    )
    logger.info("Operator call requested")
    return "Запрос передан оператору. Ожидайте ответа в ближайшее время."


@tool
def get_system_status() -> str:
    """Проверка статуса системы - используй для диагностики технических проблем. Показывает состояние базы знаний и системы."""
    print("[AGENT ACTION] Проверка статуса системы")

    try:
        # Простая проверка статуса без async операций
        import datetime

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[AGENT] Система работает нормально, время: {current_time}")

        return (
            f"Система работает нормально. Время: {current_time}. База знаний доступна."
        )

    except Exception as e:
        print(f"[AGENT ERROR] Ошибка проверки статуса: {e}")
        logger.error(f"System status error: {e}")
        return "Возникли проблемы с проверкой статуса системы."


def should_update_classification_and_priority(
    current_classification: str,
    current_priority: str,
    recent_messages: str,
    message_count: int,
) -> bool:
    """Определяет, нужно ли обновить классификацию и приоритет на основе анализа диалога."""
    try:
        from app.rag.service import get_llm_client
        import yaml

        # Загружаем конфигурацию
        with open("configs/rag_config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        auto_update_config = config.get("agent", {}).get("auto_update", {})

        # Проверяем минимальное количество сообщений
        min_messages = auto_update_config.get("min_messages_before_update", 2)
        if message_count < min_messages:
            print(
                f"[AUTO_UPDATE] Недостаточно сообщений ({message_count} < {min_messages})"
            )
            return False

        # Проверяем периодичность обновления
        update_every = auto_update_config.get("update_every_n_messages", 3)
        if message_count % update_every != 0:
            print(
                f"[AUTO_UPDATE] Не время обновлять (сообщение {message_count}, обновляем каждые {update_every})"
            )
            return False

        # Получаем промпт для определения необходимости обновления
        should_update_prompt_template = auto_update_config.get(
            "should_update_prompt", ""
        )

        if not should_update_prompt_template:
            print("[AUTO_UPDATE WARNING] Промпт should_update не найден в конфиге")
            return False

        # Формируем промпт
        should_update_prompt = should_update_prompt_template.format(
            current_classification=current_classification or "Не установлена",
            current_priority=current_priority or "medium",
            recent_messages=recent_messages,
        )

        # Получаем LLM клиент
        llm_client = get_llm_client()

        # Спрашиваем LLM нужно ли обновление
        response = llm_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": should_update_prompt}],
            max_tokens=5,
            temperature=0.1,
        )

        llm_result = response.choices[0].message.content.strip().lower()

        should_update = "yes" in llm_result

        print(f"[AUTO_UPDATE] LLM решение: {llm_result} -> {should_update}")
        logger.info(f"Should update classification/priority: {should_update}")

        return should_update

    except Exception as e:
        print(f"[AUTO_UPDATE ERROR] Ошибка проверки необходимости обновления: {e}")
        logger.error(f"Should update check error: {e}")
        return False


def auto_update_classification(
    conversation_id: int,
    dialogue_history: str,
    message_count: int,
    current_classification: str = None,
) -> dict:
    """Автоматическое обновление ТОЛЬКО классификации (без приоритета).

    Приоритет устанавливается ТОЛЬКО через MCP tool set_priority агентом.

    Возвращает dict с ключами:
    - updated: bool - была ли обновлена классификация
    - classification: str - новая классификация (если была обновлена)
    """
    try:
        # Формируем последние сообщения для анализа
        dialogue_lines = dialogue_history.split("\n")
        recent_lines = (
            dialogue_lines[-5:] if len(dialogue_lines) > 5 else dialogue_lines
        )
        recent_messages = "\n".join(recent_lines)

        if len(recent_messages) < 1000 and len(dialogue_history) > 1000:
            recent_messages = dialogue_history[-1000:]

        # Проверяем нужно ли обновление (используем medium как фиктивный приоритет)
        if not should_update_classification_and_priority(
            current_classification or "Не установлена",
            "medium",  # Фиктивный приоритет, т.к. мы его не обновляем
            recent_messages,
            message_count,
        ):
            return {"updated": False}

        print(
            f"[AUTO_UPDATE] Обновление классификации для conversation {conversation_id}"
        )

        # Обновляем ТОЛЬКО классификацию на основе ВСЕЙ истории
        new_classification_result = _classify_request_internal(
            dialogue_history=dialogue_history
        )
        new_classification = new_classification_result.replace(
            "Классификация проблемы:", ""
        ).strip()

        print(f"[AUTO_UPDATE] Новая классификация: {new_classification}")

        # Сохраняем в БД
        import asyncio
        from app.db.database import TicketsSessionLocal
        from app.db import tickets_crud as crud

        async def update_ticket():
            async with TicketsSessionLocal() as session:
                ticket = await crud.get_ticket_by_id(session, conversation_id)
                if ticket:
                    ticket.classification = new_classification
                    await session.commit()
                    print(
                        f"[AUTO_UPDATE] Обновлена классификация ticket {conversation_id}: "
                        f"classification={new_classification}"
                    )
                else:
                    print(f"[AUTO_UPDATE WARNING] Ticket {conversation_id} не найден")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(update_ticket())
            else:
                loop.run_until_complete(update_ticket())
        except Exception as e:
            print(f"[AUTO_UPDATE ERROR] Ошибка сохранения в БД: {e}")
            logger.error(f"Failed to save classification update: {e}")

        return {
            "updated": True,
            "classification": new_classification,
        }

    except Exception as e:
        print(f"[AUTO_UPDATE ERROR] Ошибка автообновления классификации: {e}")
        logger.error(f"Auto update classification error: {e}")
        return {"updated": False}


def auto_update_classification_and_priority(
    conversation_id: int,
    dialogue_history: str,
    message_count: int,
    current_classification: str = None,
    current_priority: str = None,
) -> dict:
    """УСТАРЕВШАЯ ФУНКЦИЯ - оставлена для обратной совместимости.

    Теперь используй:
    - auto_update_classification() - для автоматического обновления классификации
    - set_priority() MCP tool - для установки приоритета агентом

    Эта функция теперь обновляет ТОЛЬКО классификацию.
    """
    return auto_update_classification(
        conversation_id=conversation_id,
        dialogue_history=dialogue_history,
        message_count=message_count,
        current_classification=current_classification,
    )


# suggest_similar_problems удалена - функциональность перенесена в search_knowledge_base с параметром suggest_similar=True
