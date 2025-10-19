# Быстрый запуск с ngrok

## Для чего нужен ngrok?

Telegram бот работает через webhooks - ему нужен публичный URL для получения сообщений. 
ngrok создает туннель с вашего локального компьютера в интернет.

## Пошаговая инструкция

### 1. Подготовка .env файла

Скопируйте `.env.example` в `.env` и заполните:

```bash
# Обязательные поля
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJklmnoPQRstuVWxyz  # От @BotFather
LLM_API_KEY=ваш_ключ_для_llm                              # Для AI модели
NGROK_AUTHTOKEN=2Pf3JDFeJTHcRoi0NkCqgUVV68              # С ngrok.com

# Рекомендуемые
LLM_MODEL=gemma-2-7b                                      # Или gemma-2-9b
LLM_API_BASE=https://demo.ai.sfu-kras.ru/v1             # Ваш LLM сервер
BASE_URL=auto                                             # Автоопределение ngrok URL
```

### 2. Получение токенов

**Telegram Bot**:
1. Напишите @BotFather в Telegram
2. Команда `/newbot`
3. Придумайте имя и username бота
4. Скопируйте токен в `.env`

**ngrok Token**:
1. Зарегистрируйтесь на [ngrok.com](https://ngrok.com)
2. Dashboard → Your Authtoken
3. Скопируйте токен в `.env`

**LLM API**:
- Gemma через SFU: [demo.ai.sfu-kras.ru](https://demo.ai.sfu-kras.ru)
- OpenAI: [platform.openai.com](https://platform.openai.com)
- Другие OpenAI-совместимые API

### 3. Запуск

**Вариант A: Автоматический (Windows)**
```bash
start_with_ngrok.bat
```

**Вариант B: PowerShell (Windows)**
```powershell
.\tools\NGROK_RUNNER.ps1
```

**Вариант C: Python (любая ОС)**
```bash
python tools/run_with_ngrok.py
```

**Вариант D: Ручной запуск**
```bash
# Терминал 1: Запуск приложения
uvicorn app.main:app --reload

# Терминал 2: Запуск ngrok
ngrok http 8000
```

### 4. Результат

После запуска вы увидите:

```
ngrok public URL: https://abc123.ngrok-free.app
🌐 Application BASE_URL: https://abc123.ngrok-free.app
```

**Что работает**:
- Telegram бот: отвечает на сообщения, распознает голос
- Веб-интерфейс: `https://abc123.ngrok-free.app`
- Dashboard: `https://abc123.ngrok-free.app/dashboard`
- FAQ: `https://abc123.ngrok-free.app/faq`

## Проверка работы

1. **Telegram бот**: 
   - Найдите своего бота в Telegram
   - Напишите `/start`
   - Должно прийти приветствие со ссылкой на FAQ

2. **Веб-интерфейс**:
   - Откройте ngrok URL в браузере
   - Войдите (admin/admin по умолчанию)
   - Проверьте Dashboard и FAQ

3. **AI агент**:
   - Задайте вопрос боту: "Как подключиться к VPN?"
   - Бот должен найти ответ в базе знаний

## Troubleshooting

**Проблема**: `ngrok not found`
**Решение**: 
- Скачайте ngrok.exe в папку проекта
- Или установите глобально: `winget install ngrok`

**Проблема**: `NGROK_AUTHTOKEN not provided`
**Решение**: Добавьте токен в `.env` файл

**Проблема**: Бот не отвечает
**Решение**: 
- Проверьте `TELEGRAM_BOT_TOKEN` в `.env`
- Убедитесь что ngrok запущен и показывает URL
- Проверьте логи: должно быть "🌐 Application BASE_URL: ..."

**Проблема**: AI не работает
**Решение**:
- Проверьте `LLM_API_KEY` и `LLM_API_BASE`
- Попробуйте другую модель: `LLM_MODEL=gpt-3.5-turbo`

## Полезные команды

```bash
# Проверить статус ngrok
curl http://localhost:4040/api/tunnels

# Проверить приложение
curl http://localhost:8000/health

# Посмотреть логи
tail -f logs/app.log

# Перезапустить только приложение (ngrok оставить)
Ctrl+C в терминале с uvicorn
uvicorn app.main:app --reload
```

## Безопасность

- Не делитесь токенами (`.env` в `.gitignore`)
- ngrok URL временный - меняется при перезапуске
- Для продакшена используйте реальный домен и SSL
- Ограничьте доступ к админке (смените пароль)

---

**Готово!** Теперь у вас работает AI бот технической поддержки с публичным доступом через ngrok.