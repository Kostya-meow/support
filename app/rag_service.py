from __future__ import annotations

import json
import logging
import os
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import faiss
import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import BertForSequenceClassification, BertTokenizer
import torch

from app import tickets_crud as crud
from app.database import KnowledgeSessionLocal

logger = logging.getLogger(__name__)


def _strip_thinking_tags(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())




def _parse_float(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(match.group()) if match else None
def _preprocess(text: str, removal: set[str] | None = None) -> str:
    cleaned = re.sub(r"[^\w\s?!]", " ", text)
    words = cleaned.split()
    if removal:
        words = [w for w in words if w.lower() not in removal]
    return " ".join(words)


@dataclass
class ChatMessage:
    """Сообщение в чате для временного хранения"""
    message: str
    is_user: bool
    timestamp: datetime

@dataclass
class RAGResult:
    final_answer: str
    operator_requested: bool = False
    filter_info: dict[str, Any] | None = None
    confidence_score: float = 1.0  # Добавляем оценку уверенности (чем выше, тем хуже)


class ToxicityClassifier:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for toxicity model but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)

    def infer(self, text: str) -> float:
        inputs = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        return probs[0][1].item()


class SpeechToTextService:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        """Инициализация сервиса speech-to-text"""
        import speech_recognition as sr
        self.recognizer = sr.Recognizer()
    
    async def transcribe_audio(self, audio_file_path: str, language: str = "ru-RU") -> str:
        """
        Преобразование аудио в текст с помощью Google Speech Recognition
        
        Args:
            audio_file_path: Путь к аудио файлу
            language: Язык аудио (по умолчанию русский)
            
        Returns:
            Транскрибированный текст
        """
        import speech_recognition as sr
        import tempfile
        import os
        wav_file_path = None
        try:
            # Попытка прямого чтения (если вдруг WAV)
            try:
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data, language=language)
                return text.strip()
            except Exception as direct_error:
                logger.info(f"Direct read failed: {direct_error}, пробуем конвертацию через ffmpeg...")
                # Конвертация через ffmpeg (imageio_ffmpeg)
                try:
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception as ffmpeg_err:
                    logger.error(f"FFmpeg not found: {ffmpeg_err}")
                    return "Для распознавания голосовых сообщений требуется ffmpeg. Попробуйте отправить текст или WAV-файл."
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_temp:
                    wav_file_path = wav_temp.name
                ffmpeg_cmd = [
                    ffmpeg_path, '-y', '-i', audio_file_path,
                    '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                    wav_file_path
                ]
                import subprocess
                result = subprocess.run(ffmpeg_cmd, capture_output=True)
                if result.returncode != 0:
                    logger.error(f"FFmpeg conversion failed: {result.stderr.decode(errors='ignore')}")
                    return "Ошибка конвертации аудио. Попробуйте записать голосовое снова или отправьте текст."
                try:
                    with sr.AudioFile(wav_file_path) as source:
                        audio_data = self.recognizer.record(source)
                    text = self.recognizer.recognize_google(audio_data, language=language)
                    return text.strip()
                except Exception as recog_error:
                    logger.error(f"SpeechRecognition error after ffmpeg: {recog_error}")
                    return "Не удалось распознать голосовое сообщение. Попробуйте записать снова или отправьте текст."
        except Exception as e:
            logger.error(f"SpeechRecognition error: {e}")
            return "Не удалось распознать голосовое сообщение. Отправьте текст или WAV-файл."
        finally:
            if wav_file_path and os.path.exists(wav_file_path):
                try:
                    os.unlink(wav_file_path)
                except Exception as del_err:
                    logger.warning(f"Не удалось удалить временный файл: {del_err}")


class RAGService:
    def __init__(
        self,
        config: dict[str, Any],
    ) -> None:
        self.config = config

        llm_cfg = config.get("llm", {})

        base_url = llm_cfg.get("base_url", "")
        env_base_url = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
        if env_base_url:
            base_url = env_base_url

        api_key = llm_cfg.get("api_key", "")
        env_api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if env_api_key:
            api_key = env_api_key

        self.llm_model = llm_cfg.get("model", "")
        env_model = os.getenv("LLM_MODEL")
        if env_model:
            self.llm_model = env_model

        self.strip_thinking_tags_enabled = llm_cfg.get("strip_thinking_tags", False)

        if not self.llm_model:
            raise ValueError("LLM model name must be provided in config.llm.model or via environment variable LLM_MODEL")

        client_kwargs: dict[str, str] = {"api_key": api_key or "EMPTY"}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.llm_client = OpenAI(**client_kwargs)

        embedding_cfg = config.get("embeddings", {})
        embedding_model_name = embedding_cfg.get("model_name", "ai-forever/sbert_large_nlu_ru")
        embedding_device = embedding_cfg.get("device", "cpu")
        self.embedder = SentenceTransformer(embedding_model_name, device=embedding_device)

        toxicity_cfg = config.get("toxicity", {})
        tox_model = toxicity_cfg.get("model_path")
        if tox_model:
            self.toxicity_classifier = ToxicityClassifier(
                tox_model,
                toxicity_cfg.get("device", "cpu"),
            )
            self.toxicity_threshold = float(toxicity_cfg.get("threshold", 1.0))
        else:
            self.toxicity_classifier = None
            self.toxicity_threshold = 1.0

        rag_cfg = config.get("rag", {})
        self.top_n = int(rag_cfg.get("top_n", 20))
        self.top_m = int(rag_cfg.get("top_m", 20))
        self.top_n_tokens = int(rag_cfg.get("top_n_tokens", 250))
        self.top_m_tokens = int(rag_cfg.get("top_m_tokens", 250))
        self.filter_threshold = float(rag_cfg.get("filter_threshold", 1.0))
        self.output_threshold = float(rag_cfg.get("output_threshold", 1.0))
        self.operator_threshold = float(rag_cfg.get("operator_threshold", 0.8))
        self.history_window = int(rag_cfg.get("history_window", 3))
        self.filter_prompt = rag_cfg.get("filter_prompt", "")
        self.evaluation_prompt = rag_cfg.get("evaluation_prompt", "")
        self.persona_prompt = rag_cfg.get("persona_prompt", "")
        self.operator_intent_prompt = rag_cfg.get("operator_intent_prompt", "")

        self.filter_classification_error_message = rag_cfg.get("filter_classification_error_message", [])
        self.filter_threshold_message = rag_cfg.get("filter_threshold_message", [])
        self.toxicity_filter_message = rag_cfg.get("toxicity_filter_message", [])
        self.operator_intent_message = rag_cfg.get("operator_intent_message", [])
        self.evaluation_failure_message = rag_cfg.get("evaluation_failure_message", [])
        self.removal_list = set(rag_cfg.get("removal_list", []))

        self.histories: dict[int, list[dict[str, str]]] = defaultdict(list)
        # Хранение временной истории чатов (до создания заявки)
        self.chat_histories: dict[int, list[ChatMessage]] = defaultdict(list)
        
        # Инициализация Speech-to-Text сервиса
        self.speech_to_text = SpeechToTextService(api_key=api_key, base_url=base_url)
        
        # Кеш для саммари тикетов (простой in-memory кеш)
        self._summary_cache: dict[int, str] = {}

        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[list[str]] = []
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._faiss_matrix: np.ndarray | None = None
        self._documents: list[dict[str, str]] = []

    async def prepare(self) -> None:
        async with KnowledgeSessionLocal() as session:
            entries = await crud.load_knowledge_entries(session)
        self._load_documents(entries)

    async def reload(self) -> None:
        await self.prepare()

    def reset_history(self, conversation_id: int) -> None:
        self.histories.pop(conversation_id, None)

    def _load_documents(self, entries: Iterable[Any]) -> None:
        documents: list[dict[str, str]] = []
        vectors: list[np.ndarray] = []
        bm25_corpus: list[list[str]] = []
        for idx, entry in enumerate(entries):
            question = entry.question.strip()
            answer = entry.answer.strip()
            content = f"Вопрос: {question}\nОтвет: {answer}"
            documents.append({
                "id": idx,
                "question": question,
                "answer": answer,
                "content": content,
            })
            
            # Пропускаем записи без embedding'ов
            if entry.embedding is not None:
                vec = np.frombuffer(entry.embedding, dtype=np.float32)
                vectors.append(vec)
            
            bm25_corpus.append(_tokenize(content))
        if not documents:
            self._documents = []
            self._faiss_matrix = None
            self._faiss_index = None
            self._bm25 = None
            self._bm25_corpus = []
            logger.warning("Knowledge base is empty; RAG answers will fallback to default message.")
            return
        
        self._documents = documents
        
        # Если нет векторов (все embedding'и NULL), создаем только BM25 индекс
        if not vectors:
            self._faiss_matrix = None
            self._faiss_index = None
            logger.warning("No embeddings found; using only BM25 for search.")
        else:
            matrix = np.stack(vectors).astype("float32")
            faiss.normalize_L2(matrix)
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            self._faiss_matrix = matrix
            self._faiss_index = index
        
        self._bm25_corpus = bm25_corpus
        self._bm25 = BM25Okapi(bm25_corpus)
        logger.info("Loaded %s knowledge documents for RAG", len(documents))

    def _call_llm(self, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content.strip()
        if self.strip_thinking_tags_enabled:
            content = _strip_thinking_tags(content)
        return content

    def _check_toxicity(self, query: str) -> float:
        if not self.toxicity_classifier:
            return 0.0
        return self.toxicity_classifier.infer(query)

    def _retrieve_documents(self, query: str) -> list[dict[str, str]]:
        if not self._documents or self._faiss_index is None:
            return []
        query_vector = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype("float32")
        distances, indices = self._faiss_index.search(query_vector.reshape(1, -1), min(self.top_n, len(self._documents)))
        selected = {int(idx): float(distances[0][pos]) for pos, idx in enumerate(indices[0]) if idx != -1}

        if self._bm25 is not None and self._bm25_corpus:
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            for idx, score in enumerate(bm25_scores):
                if idx in selected:
                    selected[idx] = max(selected[idx], float(score))
                else:
                    selected[idx] = float(score)

        sorted_idx = sorted(selected.items(), key=lambda item: item[1], reverse=True)
        docs: list[dict[str, str]] = []
        total_tokens = 0
        for doc_idx, _ in sorted_idx:
            doc = self._documents[doc_idx]
            token_count = len(doc["content"].split())
            if total_tokens + token_count > self.top_n_tokens + self.top_m_tokens:
                break
            docs.append({
                "doc_id": f"doc_{doc_idx}",
                "title": doc["question"][:80] or "Документ",
                "content": doc["content"],
            })
            total_tokens += token_count
        return docs

    def _format_history(self, conversation_id: int) -> str:
        history = self.histories.get(conversation_id, [])
        if not history:
            return "История пуста."
        relevant = history[-self.history_window * 2 :]
        parts = []
        for entry in relevant:
            role = "Пользователь" if entry["role"] == "user" else "Ассистент"
            parts.append(f"{role}: {entry['content']}")
        return "\n".join(parts)

    def _apply_filter(self, user_query: str) -> tuple[str, dict[str, Any]]:
        details: dict[str, Any] = {}
        if self.filter_threshold >= 1.0 or not self.filter_prompt:
            return "", details
        messages = [
            {"role": "system", "content": self.filter_prompt},
            {"role": "user", "content": user_query},
        ]
        try:
            score_str = self._call_llm(messages, temperature=0.0, max_tokens=64)
            score = _parse_float(score_str)
            if score is None:
                raise ValueError(f'Could not parse filter score from: {score_str!r}')
            details["filter_probability"] = score
        except Exception as exc:
            logger.exception("Filter classification failed: %s", exc)
            message = random.choice(self.filter_classification_error_message) if self.filter_classification_error_message else "Не удалось классифицировать запрос."
            return message, details
        filter_score = details["filter_probability"]
        if filter_score > self.filter_threshold:
            message = random.choice(self.filter_threshold_message) if self.filter_threshold_message else "Запрос не по теме."
            return message, details
        return "", details

    def _check_operator_intent(self, user_query: str) -> tuple[bool, float]:
        if not self.operator_intent_prompt:
            return False, 0.0
        messages = [
            {"role": "system", "content": self.operator_intent_prompt},
            {"role": "user", "content": user_query},
        ]
        try:
            score_str = self._call_llm(messages, temperature=0.0, max_tokens=64)
            score = _parse_float(score_str)
            if score is None:
                raise ValueError(f'Could not parse operator score from: {score_str!r}')
        except Exception as exc:
            logger.exception("Operator intent classification failed: %s", exc)
            score = 0.0
        return score >= self.operator_threshold, score

    def _evaluate_answer(self, answer: str, dialog_text: str) -> tuple[str, float]:
        if not self.evaluation_prompt:
            return answer, 0.0
        messages = [
            {"role": "user", "content": self.evaluation_prompt},
            {"role": "user", "content": answer},
        ]
        try:
            score_str = self._call_llm(messages, temperature=0.0, max_tokens=64)
            score = _parse_float(score_str)
            if score is None:
                raise ValueError(f'Could not parse evaluation score from: {score_str!r}')
        except Exception as exc:
            logger.exception("Evaluation failed: %s", exc)
            score = math.inf
        if score > self.output_threshold:
            failure_message = random.choice(self.evaluation_failure_message) if self.evaluation_failure_message else "Ответ не прошёл проверку."
            return failure_message, score
        return answer, score

    def _store_history(self, conversation_id: int, user_text: str, answer_text: str) -> None:
        history = self.histories[conversation_id]
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": answer_text})
        if self.history_window > 0:
            max_messages = self.history_window * 2
            if len(history) > max_messages:
                self.histories[conversation_id] = history[-max_messages:]

    def add_chat_message(self, user_id: int, message: str, is_user: bool):
        """Добавляет сообщение в историю чата"""
        chat_message = ChatMessage(
            message=message,
            is_user=is_user,
            timestamp=datetime.now()
        )
        self.chat_histories[user_id].append(chat_message)
        sender_type = "USER" if is_user else "BOT"
        print(f"RAG DEBUG: Added message to user {user_id}: [{sender_type}] {message[:50]}...")
        print(f"RAG DEBUG: Total messages for user {user_id}: {len(self.chat_histories[user_id])}")

    def get_chat_history(self, user_id: int) -> list[ChatMessage]:
        """Получает историю чата для пользователя"""
        return self.chat_histories[user_id]

    def get_chat_history_since_last_ticket(self, user_id: int) -> list[ChatMessage]:
        """Получает историю чата с момента последнего закрытия заявки или с начала"""
        history = self.chat_histories[user_id]
        print(f"RAG DEBUG: Getting chat history for user {user_id}, total messages: {len(history)}")
        logging.info(f"Getting chat history for user {user_id}, total messages: {len(history)}")
        
        # Показываем всю историю для отладки
        print(f"RAG DEBUG: Full history for user {user_id}:")
        for i, msg in enumerate(history):
            sender_type = "USER" if msg.is_user else "BOT"
            print(f"  {i+1}. [{sender_type}] {msg.message[:80]}...")
        
        # Ищем последнее системное сообщение о закрытии заявки
        last_closure_index = -1
        for i in range(len(history) - 1, -1, -1):
            msg = history[i]
            if not msg.is_user and any(phrase in msg.message.lower() for phrase in [
                "завершил заявку", "заявка завершена", "если потребуется помощь"
            ]):
                last_closure_index = i
                print(f"RAG DEBUG: Found last ticket closure at index {i}: {msg.message[:50]}...")
                logging.info(f"Found last ticket closure at index {i}: {msg.message[:50]}...")
                break
        
        # Если нашли закрытие заявки, берем сообщения после него
        if last_closure_index >= 0:
            relevant_history = history[last_closure_index + 1:]
            print(f"RAG DEBUG: Using history after closure: {len(relevant_history)} messages")
            logging.info(f"Using history after closure: {len(relevant_history)} messages")
        else:
            # Если не было закрытий, берем всю историю
            relevant_history = history
            print(f"RAG DEBUG: No previous closures found, using full history: {len(relevant_history)} messages")
            logging.info(f"No previous closures found, using full history: {len(relevant_history)} messages")
        
        # Фильтруем служебные команды и системные сообщения
        filtered_history = []
        for msg in relevant_history:
            # Пропускаем служебные команды
            if msg.is_user and msg.message.startswith('/'):
                print(f"RAG DEBUG: Filtering out command: {msg.message}")
                logging.debug(f"Filtering out command: {msg.message}")
                continue
            # Пропускаем системные сообщения о создании заявки
            if not msg.is_user and any(phrase in msg.message.lower() for phrase in [
                "уведомили оператора", "ожидайте ответа", "мы уведомили"
            ]):
                print(f"RAG DEBUG: Filtering out system message: {msg.message[:30]}...")
                logging.debug(f"Filtering out system message: {msg.message[:30]}...")
                continue
            filtered_history.append(msg)
        
        print(f"RAG DEBUG: Final filtered history for user {user_id}: {len(filtered_history)} messages")
        logging.info(f"Final filtered history for user {user_id}: {len(filtered_history)} messages")
        
        # Показываем финальную отфильтрованную историю
        print(f"RAG DEBUG: Final filtered messages:")
        for i, msg in enumerate(filtered_history):
            sender_type = "USER" if msg.is_user else "BOT"
            print(f"  {i+1}. [{sender_type}] {msg.message[:80]}...")
        
        return filtered_history

    def mark_ticket_created(self, user_id: int):
        """Отмечает момент создания заявки (для будущего использования)"""
        # Можно добавить специальный маркер в историю если нужно
        pass

    def mark_ticket_closed(self, user_id: int):
        """Отмечает момент закрытия заявки"""
        # Добавляем специальное сообщение-маркер о закрытии заявки
        # Это поможет при следующем создании заявки найти точку сегментации
        closure_message = "Оператор завершил заявку. Если потребуется помощь, напишите снова или нажмите кнопку \"Позвать оператора\"."
        self.add_chat_message(user_id, closure_message, is_user=False)

    def clear_chat_history(self, user_id: int):
        """Очищает историю чата после создания заявки"""
        if user_id in self.chat_histories:
            del self.chat_histories[user_id]

    def generate_reply(self, conversation_id: int, user_message: str) -> RAGResult:
        print(f"RAG DEBUG: generate_reply called for user {conversation_id}, message: {user_message[:50]}...")
        filter_info: dict[str, Any] = {}
        query = user_message.strip()
        if not query:
            return RAGResult("Пока не вижу вопроса. Напишите подробнее?", False, filter_info, 0.0)

        # Сохраняем сообщение пользователя в историю чата
        try:
            self.add_chat_message(conversation_id, query, is_user=True)
            print(f"RAG DEBUG: Successfully saved user message")
        except Exception as e:
            print(f"RAG DEBUG: Failed to save user message: {e}")
            logging.warning(f"Failed to save user message to chat history: {e}")

        toxicity_prob = self._check_toxicity(query)
        filter_info["toxicity_probability"] = toxicity_prob
        if toxicity_prob > self.toxicity_threshold:
            message = random.choice(self.toxicity_filter_message) if self.toxicity_filter_message else "Сообщение слишком резкое. Переформулируйте, пожалуйста."
            # Сохраняем ответ бота в историю чата
            try:
                self.add_chat_message(conversation_id, message, is_user=False)
                print(f"RAG DEBUG: Successfully saved bot toxicity message")
            except Exception as e:
                print(f"RAG DEBUG: Failed to save bot toxicity message: {e}")
                logging.warning(f"Failed to save bot message to chat history: {e}")
            return RAGResult(message, False, filter_info, 0.0)

        preprocessed_query = _preprocess(query, self.removal_list)
        filter_error, filter_details = self._apply_filter(preprocessed_query)
        filter_info.update(filter_details)
        if filter_error:
            # Сохраняем ответ бота в историю чата
            try:
                self.add_chat_message(conversation_id, filter_error, is_user=False)
            except Exception as e:
                logging.warning(f"Failed to save bot message to chat history: {e}")
            return RAGResult(filter_error, False, filter_info, 0.0)

        operator_flag, operator_score = self._check_operator_intent(preprocessed_query)
        filter_info["operator_probability"] = operator_score
        if operator_flag:
            message = random.choice(self.operator_intent_message) if self.operator_intent_message else "Могу подключить оператора."
            self._store_history(conversation_id, preprocessed_query, message)
            # Сохраняем ответ бота в историю чата
            try:
                self.add_chat_message(conversation_id, message, is_user=False)
            except Exception as e:
                logging.warning(f"Failed to save bot message to chat history: {e}")
            return RAGResult(message, True, filter_info, 0.0)

        documents = self._retrieve_documents(preprocessed_query)
        if not documents:
            message = "Не нашла инструкций по этому вопросу. Попробуйте уточнить или попросите оператора."
            self._store_history(conversation_id, preprocessed_query, message)
            return RAGResult(message, False, filter_info, 1.0)  # Высокий score = низкая уверенность

        history_text = self._format_history(conversation_id)
        doc_payload = json.dumps(documents, ensure_ascii=False)
        combined_prompt = (
            f"Промпт персоны:\n{self.persona_prompt}\n\n"
            f"История диалога:\n{history_text}\n\n"
            f"Документы:\n{doc_payload}\n\n"
            f"Используя документы и историю, ответь на вопрос пользователя:\n{preprocessed_query}"
        )
        messages = [
            {"role": "system", "content": self.persona_prompt},
            {"role": "user", "content": combined_prompt},
        ]
        final_answer_raw = self._call_llm(messages, temperature=0.1, max_tokens=512)
        final_answer, eval_score = self._evaluate_answer(final_answer_raw, history_text)
        filter_info["evaluation_probability"] = eval_score
        self._store_history(conversation_id, preprocessed_query, final_answer)
        
        # Сохраняем ответ бота в историю чата
        try:
            self.add_chat_message(conversation_id, final_answer, is_user=False)
            print(f"RAG DEBUG: Successfully saved final bot answer")
        except Exception as e:
            print(f"RAG DEBUG: Failed to save final bot answer: {e}")
            logging.warning(f"Failed to save bot message to chat history: {e}")
        
        return RAGResult(final_answer, False, filter_info, eval_score)
    
    async def generate_ticket_summary(self, messages: list, ticket_id: int = None) -> str:
        """
        Генерирует краткое саммари тикета в 1-2 предложения для операторов
        
        Args:
            messages: Список сообщений тикета (объекты Message из БД)
            ticket_id: ID тикета для кеширования (опционально)
            
        Returns:
            Краткое описание проблемы
        """
        # Проверяем кеш если есть ticket_id
        if ticket_id and ticket_id in self._summary_cache:
            return self._summary_cache[ticket_id]
            
        if not messages:
            summary = "Нет сообщений в тикете"
            if ticket_id:
                self._summary_cache[ticket_id] = summary
            return summary
        
        # Собираем историю переписки
        conversation = []
        for msg in messages:
            role = "Пользователь" if msg.sender == "user" else ("Бот" if msg.sender == "bot" else "Оператор")
            conversation.append(f"{role}: {msg.text}")
        
        conversation_text = "\n".join(conversation)
        
        summary_prompt = f"""Проанализируй переписку в службе поддержки и создай краткое саммари в 1-2 предложения для оператора.
Саммари должно отражать суть проблемы пользователя и текущий статус обращения.

Переписка:
{conversation_text}

Краткое саммари:"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": summary_prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            if self.strip_thinking_tags_enabled:
                summary = _strip_thinking_tags(summary)
            
            result = summary or "Не удалось создать саммари"
            
            # Сохраняем в кеш если есть ticket_id
            if ticket_id:
                self._summary_cache[ticket_id] = result
                
            return result
            
        except Exception as e:
            logger.error(f"Error generating ticket summary: {e}")
            error_summary = "Ошибка при создании саммари"
            if ticket_id:
                self._summary_cache[ticket_id] = error_summary
            return error_summary


__all__ = ["RAGService", "RAGResult"]
