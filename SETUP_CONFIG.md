# 🚀 Быстрый старт - Настройка конфигурации

## 📋 Шаг 1: Создайте .env файл

```bash
# Скопируйте пример
cp .env.example .env

# Отредактируйте .env
nano .env  # или любой редактор
```

### Обязательные параметры в .env:

```bash
# Telegram бот (получите у @BotFather)
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# LLM API (OpenAI или совместимый)
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Секретный ключ (сгенерируйте случайную строку!)
SECRET_KEY=замените-на-случайную-строку-минимум-32-символа
```

**⚠️ ВАЖНО:**
- `.env` файл содержит секретные данные
- НЕ коммитьте `.env` в git!
- В продакшене используйте переменные окружения или secrets manager

---

## 📋 Шаг 2: Настройте config.yaml (опционально)

`config.yaml` содержит все настройки RAG системы. **По умолчанию все уже настроено!**

Редактируйте только если нужно:
- Изменить промпты
- Настроить пороги классификации
- Сменить модель эмбеддингов

**Подробное руководство:** См. `CONFIG_GUIDE.md`

---

## 📋 Шаг 3: Запустите систему

```bash
# Установите зависимости (если еще не сделали)
pip install -r requirements.txt

# Запустите web-сервер
python -m uvicorn app.main:app --reload

# В другом терминале запустите Telegram бота
python -m app.bot
```

---

## 🔐 Генерация SECRET_KEY

### Вариант 1: Python
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Вариант 2: OpenSSL
```bash
openssl rand -base64 32
```

Скопируйте результат в `.env`:
```bash
SECRET_KEY=ваш-сгенерированный-ключ
```

---

## ✅ Проверка конфигурации

### Проверить что .env загружен:
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Token:', 'OK' if os.getenv('TELEGRAM_BOT_TOKEN') else 'MISSING')"
```

### Проверить config.yaml:
```bash
python -c "from app.config import load_config; c = load_config(); print('Config loaded:', 'OK' if c else 'ERROR')"
```

---

## 🎯 Для ML-специалистов

Если вам нужно тюнить RAG пайплайн:

1. **Читайте:** `CONFIG_GUIDE.md` - полное руководство по настройке
2. **Редактируйте:** `config.yaml` - все промпты и пороги
3. **НЕ трогайте:** `.env` - только секретные данные

### Основные параметры для тюнинга:
```yaml
rag:
  filter_threshold: 1.7      # Порог фильтрации запросов
  operator_threshold: 0.8    # Порог детекции запроса оператора
  top_n: 50                  # Кол-во документов из поиска
  persona_prompt: >          # Системный промпт бота
    Ты — помощница IT-поддержки...
```

---

## 🐛 Типичные ошибки

### Ошибка: "TELEGRAM_BOT_TOKEN not found"
**Решение:** Создайте `.env` файл и добавьте токен

### Ошибка: "Config file not found"
**Решение:** Запускайте из корня проекта, где лежит `config.yaml`

### Ошибка: "Invalid API key"
**Решение:** Проверьте `LLM_API_KEY` в `.env`

---

## 📚 Документация

- `CONFIG_GUIDE.md` - Полное руководство по настройке RAG
- `.env.example` - Пример конфигурации секретов
- `config.yaml` - Конфигурация RAG системы (с комментариями)

---

## 🔄 Изменение конфигурации

### В продакшене:
1. Отредактируйте `config.yaml`
2. Перезапустите сервер: `systemctl restart support-bot`
3. Проверьте логи: `journalctl -u support-bot -f`

### При разработке:
1. Отредактируйте `config.yaml`
2. Сервер с `--reload` автоматически перезагрузится
3. Бот нужно перезапустить вручную

---

## 📞 Поддержка

При проблемах с настройкой:
1. Проверьте `.env` - все ли ключи заполнены
2. Проверьте `config.yaml` - валидный ли YAML синтаксис
3. Проверьте логи - `tail -f app.log`

**Удачного запуска! 🎉**
