# Исправления применены ✅

## Проблема 1: AttributeError 'Message' has no attribute 'read' ❌➡️✅

### Причина
В коде использовалось `msg.read` вместо правильного имени атрибута `msg.is_read`.

### Решение
**Файл**: `app/main.py` (строка 47)

Было:
```python
if msg.sender in ['user', 'bot'] and not msg.read
```

Стало:
```python
if msg.sender in ['user', 'bot'] and not msg.is_read
```

---

## Проблема 2: Счетчик не появлялся в списке заявок ❌➡️✅

### Причина
В `app/bot.py` функция `_serialize_tickets()` не добавляла поле `unread_count` к заявкам, которые отправлялись через WebSocket.

### Решение
**Файл**: `app/bot.py`

Заменили простой list comprehension на полноценный цикл с подсчетом непрочитанных:

```python
async def _serialize_tickets(session: AsyncSession) -> list[dict]:
    tickets = await crud.list_tickets(session, archived=False)
    result = []
    for ticket in tickets:
        ticket_data = TicketRead.from_orm(ticket).model_dump(mode="json")
        # Подсчитываем непрочитанные сообщения (от user и bot)
        unread_count = sum(
            1 for msg in ticket.messages 
            if msg.sender in ['user', 'bot'] and not msg.is_read
        )
        ticket_data['unread_count'] = unread_count
        result.append(ticket_data)
    return result
```

---

## Проблема 3: Звук работает только внутри чата ❌➡️✅

### Причина
Система уведомлений не могла корректно определить, нужно ли воспроизводить звук при получении нового сообщения.

### Решение

#### 1. Добавлена функция для обновления списка заявок
**Файл**: `app/main.py`

```python
async def _broadcast_conversations_update(session: AsyncSession, manager: ConnectionManager) -> None:
    """Обновить список заявок для всех подключенных клиентов"""
    tickets = await tickets_crud.list_tickets(session, archived=False)
    await manager.broadcast_conversations(_serialize_tickets(tickets))
```

Эта функция вызывается после отправки сообщения оператором, чтобы обновить список заявок:

```python
await manager.broadcast_message(conversation_id, _serialize_message(new_message))

# Обновляем список заявок для всех подключенных клиентов
await _broadcast_conversations_update(session, manager)
```

#### 2. Улучшена логика уведомлений
**Файл**: `app/static/notification.js`

Теперь звук воспроизводится в двух случаях:

1. **При увеличении счетчика непрочитанных** (обновление списка заявок):
```javascript
if (payload.type === 'conversations' && payload.conversations) {
    let shouldPlaySound = false;
    
    payload.conversations.forEach(conversation => {
        const prevCount = lastUnreadCounts[conversation.id] || 0;
        const newCount = conversation.unread_count || 0;
        
        if (newCount > prevCount) {
            shouldPlaySound = true;
        }
        
        lastUnreadCounts[conversation.id] = newCount;
    });
    
    if (shouldPlaySound) {
        playNotificationSound();
    }
}
```

2. **При получении сообщения напрямую** (если ты НЕ в этом чате):
```javascript
if (payload.type === 'message' && payload.message) {
    const currentPath = window.location.pathname;
    const isInChat = currentPath.includes('/conversations/');
    const isFromUser = payload.message.sender === 'user' || payload.message.sender === 'bot';
    
    if (isFromUser) {
        const messageConvId = payload.conversation_id || payload.message.conversation_id;
        const currentConvId = currentPath.match(/\/conversations\/(\d+)/)?.[1];
        
        // Играем звук, если мы не в этом конкретном чате
        if (!isInChat || currentConvId !== String(messageConvId)) {
            playNotificationSound();
        }
    }
}
```

---

## Как это работает теперь

### Сценарий 1: Пользователь отправляет сообщение в Telegram
1. Бот получает сообщение и сохраняет его в БД
2. Бот вызывает `_broadcast_message()` - отправляет сообщение в чат WebSocket
3. Бот вызывает `_broadcast_tickets()` - обновляет список заявок с новым `unread_count`
4. notification.js видит увеличение счетчика и воспроизводит звук
5. В UI счетчик отображается: `Заявка #28 (5)`

### Сценарий 2: Оператор отправляет сообщение пользователю
1. API получает сообщение от оператора
2. Сообщение сохраняется в БД
3. Вызывается `broadcast_message()` - отправка в чат
4. Вызывается `_broadcast_conversations_update()` - обновление списка
5. Если есть другие операторы на других страницах - они получают обновление

### Сценарий 3: Оператор на странице Dashboard
1. notification.js подключается к `/ws/conversations`
2. При новом сообщении получает обновление со счетчиком
3. Видит увеличение `unread_count` и воспроизводит звук
4. Оператор слышит уведомление и переходит в чат

---

## ⚠️ ВАЖНО: Требуется перезапуск сервера

Изменения требуют перезапуска:

```bash
# Остановить сервер (Ctrl+C)
# Запустить снова
.venv\Scripts\activate
python -m uvicorn app.main:app --reload
```

---

## Проверка работы

### ✅ Ошибка AttributeError исчезла
Больше не будет ошибки `'Message' object has no attribute 'read'`

### ✅ Счетчик отображается
В списке заявок показывается: `Заявка #28 (5)` где 5 - количество непрочитанных

### ✅ Звук работает глобально
- Откройте Dashboard
- Попросите кого-то написать в Telegram
- Звук должен воспроизвестись, даже если вы не в чате

### ✅ Логи в консоли (F12)
```
🔔 Global notification system initializing...
Global notification audio initialized
✅ Global notifications WebSocket connected
📦 WebSocket payload: conversations
🔔 New unread messages in conversation #28: 0 -> 1
🔔 Notification sound played
```

---

## Файлы изменены
- ✅ `app/main.py` - исправлен `msg.read` → `msg.is_read`, добавлена функция `_broadcast_conversations_update()`
- ✅ `app/bot.py` - функция `_serialize_tickets()` теперь добавляет `unread_count`
- ✅ `app/static/notification.js` - улучшена логика определения когда воспроизводить звук
- ✅ `app/tickets_crud.py` - добавлен `selectinload()` (из предыдущего фикса)

---

## Техническая справка

### Почему раньше не работало?
1. **bot.py не отправлял unread_count** - список заявок приходил без счетчиков
2. **notification.js не мог правильно определить новые сообщения** - полагался только на изменение счетчика
3. **Обновления не приходили на другие страницы** - оператор в чате получал обновления, но другие операторы нет

### Что изменилось?
1. **Все сериализаторы считают unread_count** (и в main.py, и в bot.py)
2. **WebSocket обновляет список после каждого сообщения** (и от бота, и от оператора)
3. **notification.js имеет двойную логику** (отслеживание счетчика + прямые сообщения)
4. **Звук не воспроизводится в активном чате** (чтобы не раздражать когда общаешься)
