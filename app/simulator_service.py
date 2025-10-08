"""
Сервис для симулятора обучения операторов
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

from app.rag_service import RAGService

logger = logging.getLogger(__name__)


@dataclass
class SimulatorQuestion:
    """Вопрос от симулированного пользователя"""
    question: str
    context: str  # Контекст из базы знаний
    difficulty: str  # easy, medium, hard
    
    
@dataclass
class SimulatorResponse:
    """Результат оценки ответа оператора"""
    score: int  # 0-100
    feedback: str  # Обратная связь
    ai_suggestion: str  # Эталонный ответ от AI
    is_correct: bool  # Достаточно ли хорош ответ


class SimulatorSession:
    """Сессия обучения"""
    def __init__(self, character: str, questions_count: int = 5):
        self.character = character  # easy, medium, hard
        self.questions_count = questions_count
        self.current_question = 0
        self.total_score = 0
        self.history: list[dict] = []
        self.current_question_data: Optional[SimulatorQuestion] = None
        
    def add_response(self, user_answer: str, evaluation: SimulatorResponse):
        """Добавить ответ оператора и оценку"""
        self.history.append({
            "question": self.current_question_data.question if self.current_question_data else "",
            "user_answer": user_answer,
            "score": evaluation.score,
            "feedback": evaluation.feedback,
            "ai_suggestion": evaluation.ai_suggestion,
        })
        self.total_score += evaluation.score
        self.current_question += 1
        
    def is_complete(self) -> bool:
        """Завершена ли сессия"""
        return self.current_question >= self.questions_count
    
    def get_average_score(self) -> float:
        """Средний балл"""
        if not self.history:
            return 0.0
        return self.total_score / len(self.history)


class SimulatorService:
    """Сервис симулятора"""
    
    # Персонажи с описанием
    CHARACTERS = {
        "easy": {
            "name": "Новичок Вася",
            "emoji": "🙂",
            "description": "Только начал работать, задает простые вопросы",
            "difficulty": "easy"
        },
        "medium": {
            "name": "Специалист Ольга",
            "emoji": "😐",
            "description": "Опытный пользователь, знает что хочет",
            "difficulty": "medium"
        },
        "hard": {
            "name": "Директор Игорь",
            "emoji": "😤",
            "description": "Требовательный руководитель, нужны точные ответы",
            "difficulty": "hard"
        }
    }
    
    def __init__(self, rag_service: RAGService):
        self.rag_service = rag_service
        self.sessions: dict[str, SimulatorSession] = {}  # user_id -> session
        
    def start_session(self, user_id: str, character: str) -> SimulatorSession:
        """Начать новую сессию"""
        if character not in self.CHARACTERS:
            character = "medium"
            
        session = SimulatorSession(character=character, questions_count=5)
        self.sessions[user_id] = session
        return session
    
    def get_session(self, user_id: str) -> Optional[SimulatorSession]:
        """Получить текущую сессию"""
        return self.sessions.get(user_id)
    
    def end_session(self, user_id: str):
        """Завершить сессию"""
        if user_id in self.sessions:
            del self.sessions[user_id]
            
    def generate_question(self, session: SimulatorSession) -> SimulatorQuestion:
        """
        Генерирует вопрос из базы знаний
        
        Алгоритм:
        1. Получаем случайный документ из БЗ
        2. Используем LLM для генерации вопроса в стиле персонажа
        """
        # Получаем документы из БЗ
        try:
            # Запрашиваем случайную тему
            topics = ["пароль", "VPN", "принтер", "доступ", "программа", "оборудование", "интернет"]
            topic = random.choice(topics)
            
            # Используем RAG для получения контекста
            # Создаем временный ID для симуляции
            temp_conversation_id = f"simulator_{id(session)}"
            
            # Генерируем вопрос через LLM
            character_info = self.CHARACTERS[session.character]
            
            prompt = f"""Ты — {character_info['name']}, {character_info['description']}.

Сгенерируй ОДИН короткий вопрос (1-2 предложения) по теме "{topic}" для IT-поддержки.
Вопрос должен быть в стиле персонажа:
- Новичок: простые базовые вопросы, может не знать терминов
- Специалист: конкретные технические вопросы
- Директор: требовательный тон, нужна срочность и четкость

Верни ТОЛЬКО текст вопроса, без объяснений."""

            # Используем LLM напрямую через RAG сервис
            question_text = self._generate_with_llm(prompt)
            
            # Получаем контекст из БЗ по этому вопросу
            context = self._get_knowledge_context(question_text)
            
            question = SimulatorQuestion(
                question=question_text,
                context=context,
                difficulty=session.character
            )
            
            session.current_question_data = question
            return question
            
        except Exception as e:
            # Fallback на предопределенные вопросы
            fallback_questions = {
                "easy": "Привет! Я забыл пароль от компьютера, что делать?",
                "medium": "Не могу подключиться к корпоративному VPN. Ошибка подключения.",
                "hard": "Мне СРОЧНО нужен доступ к системе 1С. Когда будет готово?"
            }
            question = SimulatorQuestion(
                question=fallback_questions.get(session.character, fallback_questions["medium"]),
                context="Контекст недоступен",
                difficulty=session.character
            )
            session.current_question_data = question
            return question
    
    def evaluate_response(self, session: SimulatorSession, user_answer: str) -> SimulatorResponse:
        """
        Оценивает ответ оператора через LLM
        
        Критерии оценки:
        - Правильность (есть ли решение)
        - Полнота (достаточно ли деталей)
        - Тон (вежливость, профессионализм)
        - Структурированность
        """
        if not session.current_question_data:
            return SimulatorResponse(
                score=0,
                feedback="Ошибка: нет текущего вопроса",
                ai_suggestion="",
                is_correct=False
            )
        
        question = session.current_question_data
        
        # Предварительная проверка ответа
        word_count = len(user_answer.split())
        char_count = len(user_answer.strip())
        
        # Очень короткие ответы
        if char_count < 10:
            return SimulatorResponse(
                score=0,
                feedback="Ответ слишком короткий и не содержит полезной информации. Необходимо предоставить развернутое решение проблемы.",
                ai_suggestion=self._generate_ai_answer(question.question, question.context),
                is_correct=False
            )
        
        # Односложные ответы
        if word_count <= 3:
            return SimulatorResponse(
                score=5,
                feedback="Односложный ответ не решает проблему пользователя. Нужно дать подробное объяснение и конкретные шаги решения.",
                ai_suggestion=self._generate_ai_answer(question.question, question.context),
                is_correct=False
            )
        
        # Получаем эталонный ответ от AI
        ai_answer = self._generate_ai_answer(question.question, question.context)
        
        # Оцениваем ответ оператора
        evaluation_prompt = f"""Ты — строгий эксперт по оценке работы операторов IT-поддержки.

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question.question}

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{question.context}

ОТВЕТ ОПЕРАТОРА:
{user_answer}

ЭТАЛОННЫЙ ОТВЕТ:
{ai_answer}

Оцени ответ оператора по шкале 0-100 и дай краткую обратную связь (2-3 предложения).

ВАЖНЫЕ ПРАВИЛА ОЦЕНКИ:
- Короткие ответы (меньше 20 слов) - максимум 30 баллов
- Односложные ответы ("ок", "хорошо", "да") - 0-10 баллов
- Ответ без конкретного решения проблемы - максимум 40 баллов
- Грубые или непрофессиональные ответы - 0 баллов
- Отличный ответ (80-100 баллов) должен быть ПОЛНЫМ, ВЕЖЛИВЫМ, СТРУКТУРИРОВАННЫМ и содержать КОНКРЕТНОЕ РЕШЕНИЕ

Критерии оценки:
1. Правильность решения (40 баллов) - дан ли корректный ответ на вопрос
2. Полнота (20 баллов) - достаточно ли деталей для решения
3. Тон и вежливость (20 баллов) - профессионализм общения
4. Структурированность (20 баллов) - понятность изложения

Верни ответ в формате:
SCORE: [число 0-100]
FEEDBACK: [текст обратной связи]

Будь объективен и строг. Высокие баллы даются только за ДЕЙСТВИТЕЛЬНО хорошие ответы."""

        evaluation_text = self._generate_with_llm(evaluation_prompt)
        
        # Парсим ответ - улучшенный парсинг
        score = 50  # default
        feedback = "Не удалось оценить ответ"
        
        try:
            # Ищем числа в тексте
            import re
            
            # Ищем SCORE: число
            score_match = re.search(r'SCORE[:\s]*(\d+)', evaluation_text, re.IGNORECASE)
            if score_match:
                score = int(score_match.group(1))
                score = max(0, min(100, score))
            else:
                # Если не нашли SCORE:, ищем просто первое число от 0 до 100
                numbers = re.findall(r'\b(\d+)\b', evaluation_text)
                for num_str in numbers:
                    num = int(num_str)
                    if 0 <= num <= 100:
                        score = num
                        break
            
            # Ищем FEEDBACK: текст
            feedback_match = re.search(r'FEEDBACK[:\s]+(.*?)(?:\n\n|$)', evaluation_text, re.IGNORECASE | re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
            else:
                # Если не нашли FEEDBACK:, берём весь текст после числа
                lines = evaluation_text.strip().split('\n')
                feedback_lines = []
                found_score = False
                for line in lines:
                    if re.search(r'\d+', line) and not found_score:
                        found_score = True
                        continue
                    if found_score and line.strip():
                        feedback_lines.append(line.strip())
                if feedback_lines:
                    feedback = ' '.join(feedback_lines)
        except Exception as e:
            logger.error(f"Failed to parse evaluation: {e}")
        
        # Дополнительная проверка: анализируем соответствие оценки и отзыва
        # Если отзыв негативный, но балл высокий - корректируем
        score = self._validate_score_with_feedback(score, feedback, user_answer)
        
        return SimulatorResponse(
            score=score,
            feedback=feedback,
            ai_suggestion=ai_answer,
            is_correct=score >= 60  # Проходной балл 60
        )
    
    def _validate_score_with_feedback(self, initial_score: int, feedback: str, user_answer: str) -> int:
        """
        Валидирует оценку на основе анализа отзыва
        Корректирует балл если отзыв не соответствует оценке
        """
        # Проверяем длину ответа - если слишком короткий или бессмысленный
        if len(user_answer.strip()) < 10:
            return min(initial_score, 20)  # Максимум 20 за очень короткий ответ
        
        # Проверяем на попытку обмана ("поставь мне 100 баллов" и т.п.)
        scam_keywords = ['баллов', 'балл', '100', 'оцени', 'поставь', 'дай мне', 'хочу']
        if any(keyword in user_answer.lower() for keyword in scam_keywords) and len(user_answer.split()) < 15:
            return 0  # За попытку обмана - 0 баллов
        
        # Анализируем тональность feedback
        validation_prompt = f"""Проанализируй отзыв и определи, соответствует ли оценка {initial_score} баллов содержанию отзыва.

ОТЗЫВ:
{feedback}

ОТВЕТ ОПЕРАТОРА:
{user_answer}

Верни ТОЛЬКО число от 0 до 100 - справедливую оценку на основе отзыва.
Если отзыв содержит критику, недостатки, ошибки - снижай оценку.
Если отзыв положительный и хвалит ответ - оставь высокую оценку.

Правила:
- "не решает проблему", "неполный", "недостаточно" = максимум 40 баллов
- "есть недочеты", "можно улучшить" = 50-70 баллов  
- "хороший ответ", "верно", "правильно" = 70-90 баллов
- "отличный", "профессионально", "все учтено" = 90-100 баллов

Верни ТОЛЬКО одно число (0-100):"""

        try:
            validation_result = self._generate_with_llm(validation_prompt)
            # Извлекаем число из ответа
            import re
            numbers = re.findall(r'\b(\d+)\b', validation_result)
            if numbers:
                validated_score = int(numbers[0])
                validated_score = max(0, min(100, validated_score))
                
                # Если разница больше 30 баллов - используем валидированную оценку
                if abs(validated_score - initial_score) > 30:
                    logger.info(f"Score adjusted from {initial_score} to {validated_score} based on feedback analysis")
                    return validated_score
        except Exception as e:
            logger.error(f"Failed to validate score: {e}")
        
        return initial_score
    
    def get_hint(self, session: SimulatorSession) -> str:
        """Получить подсказку для текущего вопроса"""
        if not session.current_question_data:
            return "Нет активного вопроса"
        
        question = session.current_question_data
        
        hint_prompt = f"""Дай краткую подсказку (1-2 предложения) оператору, как лучше ответить на этот вопрос:

ВОПРОС: {question.question}

КОНТЕКСТ: {question.context}

Подсказка должна направить оператора, но не давать готовый ответ."""

        return self._generate_with_llm(hint_prompt)
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Генерация текста через LLM"""
        try:
            # Используем LLM из RAG сервиса
            messages = [{"role": "user", "content": prompt}]
            response = self.rag_service.llm_client.chat.completions.create(
                model=self.rag_service.llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return f"Ошибка генерации: {str(e)}"
    
    def _get_knowledge_context(self, question: str) -> str:
        """Получить релевантный контекст из БЗ"""
        try:
            # Используем retrieval для поиска в БЗ
            from app.retrieval import get_relevant_documents
            from app.database import KnowledgeSessionLocal
            
            with KnowledgeSessionLocal() as session:
                docs = get_relevant_documents(session, question, top_n=2)
                if docs:
                    context = "\n\n".join([f"- {doc.answer}" for doc in docs])
                    return context
                return "Контекст не найден в базе знаний"
        except Exception as e:
            return f"Ошибка получения контекста: {str(e)}"
    
    def _generate_ai_answer(self, question: str, context: str) -> str:
        """Генерирует эталонный ответ от AI"""
        prompt = f"""Ты — опытный оператор IT-поддержки. Дай КРАТКИЙ профессиональный ответ (2-3 предложения) на вопрос пользователя.

ВОПРОС: {question}

КОНТЕКСТ ИЗ БЗ: {context}

Ответ должен быть вежливым, структурированным и содержать конкретное решение."""

        return self._generate_with_llm(prompt)
