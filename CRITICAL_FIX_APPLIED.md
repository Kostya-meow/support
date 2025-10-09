# Критические исправления применены ✅

## Проблема 1: Ошибка SQLAlchemy MissingGreenlet ❌➡️✅

### Причина
При попытке доступа к `ticket.messages` в функции `_serialize_tickets()` происходила ленивая загрузка (lazy loading) связанных данных в асинхронном контексте WebSocket, что вызывало ошибку `MissingGreenlet`.

### Решение
Добавлена жадная загрузка (eager loading) сообщений в запросе `list_tickets()`:

**Файл**: `app/tickets_crud.py`

```python
# Добавлен импорт
from sqlalchemy.orm import selectinload

# Изменена строка в функции list_tickets()
stmt = select(models.Ticket).options(selectinload(models.Ticket.messages)).order_by(...)
```

Теперь при получении списка заявок все сообщения загружаются заранее, и обращение к `ticket.messages` в `_serialize_tickets()` не вызывает дополнительных SQL-запросов.

---

## Проблема 2: Звук работает только в чате ❌➡️✅

### Причина
Файл `notification.js` проверял наличие `window.userPermissions`, которая не была установлена на всех страницах, и отключал систему уведомлений.

### Решение
Удалена проверка прав доступа из `notification.js` - теперь система уведомлений работает на всех страницах для всех авторизованных пользователей.

**Файл**: `app/static/notification.js`

Удалено:
```javascript
const userPermissions = window.userPermissions || [];
const hasTicketsAccess = userPermissions.includes('tickets') || userPermissions.includes('requests');

if (!hasTicketsAccess) {
    console.log('No tickets access - notifications disabled');
    return;
}
```

---

## Как работает система теперь

### Глобальные уведомления
1. **notification.js** загружается на всех страницах
2. Автоматически подключается к WebSocket `/ws/conversations`
3. Отслеживает изменения счетчика непрочитанных сообщений (`unread_count`)
4. При увеличении счетчика воспроизводит звук уведомления

### Счетчик непрочитанных
1. Сервер загружает все сообщения вместе со списком заявок (eager loading)
2. Функция `_serialize_tickets()` считает непрочитанные:
   ```python
   unread_count = sum(
       1 for msg in ticket.messages 
       if not msg.is_read and msg.sender != 'operator'
   )
   ```
3. Счетчик передается через WebSocket всем подключенным клиентам
4. В UI отображается рядом с номером заявки: `Заявка #28 (5)`

---

## ⚠️ ВАЖНО: Требуется перезапуск сервера

Чтобы исправления вступили в силу, необходимо перезапустить сервер:

### Вариант 1: Через батник
```
restart_server.bat
```

### Вариант 2: Вручную
1. Остановить текущий процесс uvicorn (Ctrl+C в терминале)
2. Запустить заново:
```
.venv\Scripts\activate
python -m uvicorn app.main:app --reload
```

---

## Проверка работы

После перезапуска:

1. ✅ Ошибка `MissingGreenlet` больше не должна появляться в логах
2. ✅ WebSocket `/ws/conversations` должен успешно подключаться
3. ✅ В списке заявок должны отображаться счетчики: `Заявка #X (Y)`
4. ✅ При получении нового сообщения звук должен воспроизводиться **на любой странице**, не только в чате

### Проверка в консоли браузера (F12)
Должны появиться сообщения:
```
🔔 Global notification system initializing...
Global notification audio initialized
✅ Global notifications WebSocket connected
```

При получении нового сообщения:
```
📦 WebSocket payload: conversations
🔔 New unread messages in conversation #28: 0 -> 1
🔔 Notification sound played
```

---

## Файлы изменены
- ✅ `app/tickets_crud.py` - добавлен `selectinload()`
- ✅ `app/static/notification.js` - убрана проверка прав
