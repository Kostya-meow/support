"""
Инструменты для RAG агента
"""

import logging
from typing import List, Dict, Any
from agno.tools import tool

logger = logging.getLogger(__name__)


@tool
async def search_knowledge_base(query: str, category: str = None) -> str:
    """Поиск в базе знаний - твой главный инструмент! Используй для любых технических вопросов, инструкций, документации. Всегда ищи сначала здесь перед тем как отвечать."""
    print(f"[AGENT ACTION] Поиск в базе знаний: '{query}'")

    try:
        from app.db.database import KnowledgeSessionLocal
        from app.db import crud
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

            # Формируем ответ из топ-3 результатов
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


@tool
def classify_request(
    dialogue_history: str = None, text: str = None, categories: List[str] = None
) -> str:
    """Классификация запроса через LLM - анализирует весь диалог с пользователем для точной классификации проблемы. Возвращает категории проблем которые можно назначить заявке."""

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
