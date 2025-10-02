from __future__ import annotations

import asyncio
from typing import Iterable, Optional, Sequence

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app import crud, models

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class KnowledgeBase:
    def __init__(self, session_maker: async_sessionmaker[AsyncSession], model_name: str = DEFAULT_MODEL_NAME) -> None:
        self._session_maker = session_maker
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._entries: list[dict] = []
        self._dim: int | None = None
        self._state_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()

    async def ensure_loaded(self) -> None:
        async with self._session_maker() as session:
            entries = await crud.load_knowledge_entries(session)
        await self._update_state_from_entries(entries)

    async def rebuild_from_pairs(self, pairs: Sequence[tuple[str, str]]) -> int:
        pairs = [(q.strip(), a.strip()) for q, a in pairs if q and a]
        if not pairs:
            async with self._session_maker() as session:
                await crud.replace_knowledge_entries(session, [])
            await self._update_state([], None)
            return 0

        model = await self._get_model()
        questions = [question for question, _ in pairs]
        answers = [answer for _, answer in pairs]
        embeddings = await _encode_texts(model, questions)

        serialized = [embedding.tobytes() for embedding in embeddings]
        async with self._session_maker() as session:
            await crud.replace_knowledge_entries(session, zip(questions, answers, serialized))
            entries = await crud.load_knowledge_entries(session)

        await self._update_state_from_entries(entries)
        return len(entries)

    async def search(self, query: str, top_k: int = 1) -> Optional[dict]:
        model = await self._get_model()
        vector = await _encode_texts(model, [query])

        async with self._state_lock:
            if self._index is None or not self._entries:
                return None
            index = self._index
            entries = list(self._entries)

        k = min(top_k, len(entries))
        distances, indices = index.search(vector, k)
        best_idx = int(indices[0][0]) if indices.size else -1
        if best_idx < 0 or best_idx >= len(entries):
            return None
        result = entries[best_idx].copy()
        result["score"] = float(distances[0][0])
        return result

    async def _update_state_from_entries(self, entries: Iterable[models.KnowledgeEntry]) -> None:
        entries = list(entries)
        if not entries:
            await self._update_state([], None)
            return

        vectors = []
        simplified_entries: list[dict] = []
        dim: int | None = None
        for entry in entries:
            # Добавляем все записи в simplified_entries
            simplified_entries.append(
                {
                    "id": entry.id,
                    "question": entry.question,
                    "answer": entry.answer,
                }
            )
            
            # Добавляем векторы только для записей с embedding'ами
            if entry.embedding is not None:
                array = np.frombuffer(entry.embedding, dtype=np.float32)
                if dim is None:
                    dim = array.shape[0]
                else:
                    if array.shape[0] != dim:
                        raise ValueError("Inconsistent embedding dimensions in database")
                vectors.append(array)

        # Если нет embedding'ов, создаем пустой индекс
        if not vectors:
            await self._update_state(simplified_entries, None)
            return

        matrix = np.stack(vectors).astype("float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(dim or matrix.shape[1])
        index.add(matrix)

        await self._update_state(simplified_entries, index)

    async def _update_state(self, entries: list[dict], index: faiss.Index | None) -> None:
        async with self._state_lock:
            self._entries = entries
            self._index = index
            self._dim = index.d if index is not None else None

    async def _get_model(self) -> SentenceTransformer:
        async with self._model_lock:
            if self._model is None:
                loop = asyncio.get_running_loop()
                self._model = await loop.run_in_executor(None, SentenceTransformer, self._model_name)
        return self._model


async def _encode_texts(model: SentenceTransformer, texts: Sequence[str]) -> np.ndarray:
    loop = asyncio.get_running_loop()
    embeddings = await loop.run_in_executor(
        None,
        lambda: model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    faiss.normalize_L2(embeddings)
    return embeddings
