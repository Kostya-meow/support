"""
Миграция: Изменение telegram_chat_id на TEXT и добавление vk_message_id
"""
import sqlite3

def migrate():
    # Подключаемся к базе tickets
    conn = sqlite3.connect('tickets.db')
    cursor = conn.cursor()

    try:
        # Проверяем текущую структуру таблицы tickets
        cursor.execute("PRAGMA table_info(tickets)")
        ticket_columns = {column[1]: column for column in cursor.fetchall()}

        # Проверяем текущую структуру таблицы messages
        cursor.execute("PRAGMA table_info(messages)")
        message_columns = {column[1]: column for column in cursor.fetchall()}

        changes_needed = False

        # Проверяем telegram_chat_id в tickets
        if 'telegram_chat_id' in ticket_columns:
            col_type = ticket_columns['telegram_chat_id'][2].upper()
            if 'TEXT' not in col_type and 'VARCHAR' not in col_type:
                changes_needed = True
                print(f"Текущий тип telegram_chat_id: {col_type}, нужно изменить на TEXT")

        # Проверяем vk_message_id в messages
        if 'vk_message_id' not in message_columns:
            changes_needed = True
            print("Добавляем колонку vk_message_id...")

        if not changes_needed:
            print("⚠️  Миграция не требуется, все изменения уже применены")
            return

        # Создаем временные таблицы с новой структурой
        print("Создаем временные таблицы с новой структурой...")

        # Новая структура tickets
        cursor.execute("""
            CREATE TABLE tickets_new (
                id INTEGER PRIMARY KEY,
                telegram_chat_id TEXT NOT NULL,
                title VARCHAR(255),
                summary TEXT,
                status VARCHAR(20) NOT NULL DEFAULT 'open',
                priority VARCHAR(10) NOT NULL DEFAULT 'medium',
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                first_response_at DATETIME,
                closed_at DATETIME,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Новая структура messages
        cursor.execute("""
            CREATE TABLE messages_new (
                id INTEGER PRIMARY KEY,
                ticket_id INTEGER NOT NULL,
                sender VARCHAR(32) NOT NULL,
                text TEXT NOT NULL,
                telegram_message_id INTEGER,
                vk_message_id INTEGER,
                is_system BOOLEAN NOT NULL DEFAULT 0,
                is_read BOOLEAN NOT NULL DEFAULT 0,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticket_id) REFERENCES tickets_new (id)
            )
        """)

        # Копируем данные
        print("Копируем данные...")

        # Копируем tickets (telegram_chat_id преобразуем в строку)
        cursor.execute("""
            INSERT INTO tickets_new (id, telegram_chat_id, title, summary, status, priority, created_at, first_response_at, closed_at, updated_at)
            SELECT id, CAST(telegram_chat_id AS TEXT), title, summary, status, priority, created_at, first_response_at, closed_at, updated_at
            FROM tickets
        """)

        # Копируем messages
        cursor.execute("""
            INSERT INTO messages_new (id, ticket_id, sender, text, telegram_message_id, is_system, is_read, created_at)
            SELECT id, ticket_id, sender, text, telegram_message_id, is_system, is_read, created_at
            FROM messages
        """)

        # Заменяем таблицы
        print("Заменяем таблицы...")
        cursor.execute("DROP TABLE messages")
        cursor.execute("DROP TABLE tickets")

        cursor.execute("ALTER TABLE tickets_new RENAME TO tickets")
        cursor.execute("ALTER TABLE messages_new RENAME TO messages")

        # Создаем индексы
        cursor.execute("CREATE INDEX ix_tickets_telegram_chat_id ON tickets (telegram_chat_id)")
        cursor.execute("CREATE INDEX ix_messages_ticket_id ON messages (ticket_id)")

        conn.commit()
        print("✅ Миграция успешно выполнена!")
        print("   - telegram_chat_id изменен на TEXT")
        print("   - Добавлена колонка vk_message_id")

    except Exception as e:
        print(f"❌ Ошибка миграции: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()