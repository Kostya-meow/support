#!/usr/bin/env python3
"""
Миграция с старой структуры (conversations) на новую (tickets)
"""

import asyncio
import sqlite3
from pathlib import Path
from datetime import datetime

async def migrate_to_tickets():
    """Мигрирует данные из старой структуры в новую."""
    
    old_db_path = Path("support.db")
    tickets_db_path = Path("tickets.db") 
    knowledge_db_path = Path("knowledge.db")
    
    if not old_db_path.exists():
        print("❌ Старая база данных support.db не найдена")
        return
    
    # Удаляем новые БД если они уже существуют
    for db_path in [tickets_db_path, knowledge_db_path]:
        if db_path.exists():
            db_path.unlink()
            print(f"🗑️  Удалена существующая БД: {db_path}")
    
    # Подключаемся к старой БД
    old_conn = sqlite3.connect(old_db_path)
    old_cursor = old_conn.cursor()
    
    # Создаем новые БД
    tickets_conn = sqlite3.connect(tickets_db_path)
    tickets_cursor = tickets_conn.cursor()
    
    knowledge_conn = sqlite3.connect(knowledge_db_path)
    knowledge_cursor = knowledge_conn.cursor()
    
    try:
        # Создаем таблицы в новых БД
        print("📋 Создание таблиц...")
        
        # Таблица заявок
        tickets_cursor.execute('''
            CREATE TABLE tickets (
                id INTEGER PRIMARY KEY,
                telegram_chat_id BIGINT NOT NULL,
                title VARCHAR(255),
                status VARCHAR(20) NOT NULL DEFAULT 'open',
                priority VARCHAR(10) NOT NULL DEFAULT 'medium',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                closed_at TIMESTAMP
            )
        ''')
        
        tickets_cursor.execute('CREATE INDEX ix_tickets_telegram_chat_id ON tickets (telegram_chat_id)')
        
        # Таблица сообщений
        tickets_cursor.execute('''
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                ticket_id INTEGER NOT NULL,
                sender VARCHAR(32) NOT NULL,
                text TEXT NOT NULL,
                telegram_message_id BIGINT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticket_id) REFERENCES tickets (id)
            )
        ''')
        
        tickets_cursor.execute('CREATE INDEX ix_messages_ticket_id ON messages (ticket_id)')
        
        # Таблица базы знаний
        knowledge_cursor.execute('''
            CREATE TABLE knowledge_entries (
                id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Мигрируем conversations -> tickets
        print("🎫 Миграция диалогов в заявки...")
        old_cursor.execute("SELECT * FROM conversations")
        conversations = old_cursor.fetchall()
        
        # Получаем названия колонок
        old_cursor.execute("PRAGMA table_info(conversations)")
        columns_info = old_cursor.fetchall()
        col_names = [col[1] for col in columns_info]
        
        conversation_mapping = {}  # old_id -> new_id
        
        for conv_data in conversations:
            conv_dict = dict(zip(col_names, conv_data))
            
            # Определяем статус заявки
            status = 'archived'
            if 'is_archived' in conv_dict and not conv_dict['is_archived']:
                if 'operator_requested' in conv_dict and conv_dict['operator_requested']:
                    status = 'open'
                else:
                    status = 'closed'
            
            # Вставляем заявку
            tickets_cursor.execute('''
                INSERT INTO tickets (telegram_chat_id, title, status, created_at, updated_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                conv_dict['telegram_chat_id'],
                conv_dict.get('title'),
                status,
                conv_dict['created_at'],
                conv_dict['updated_at'],
                conv_dict['updated_at'] if status in ['closed', 'archived'] else None
            ))
            
            new_ticket_id = tickets_cursor.lastrowid
            conversation_mapping[conv_dict['id']] = new_ticket_id
        
        print(f"✅ Мигрировано заявок: {len(conversations)}")
        
        # Мигрируем messages
        print("💬 Миграция сообщений...")
        old_cursor.execute("SELECT * FROM messages")
        messages = old_cursor.fetchall()
        
        old_cursor.execute("PRAGMA table_info(messages)")
        msg_columns_info = old_cursor.fetchall()
        msg_col_names = [col[1] for col in msg_columns_info]
        
        migrated_messages = 0
        for msg_data in messages:
            msg_dict = dict(zip(msg_col_names, msg_data))
            
            old_conv_id = msg_dict.get('conversation_id')
            if old_conv_id in conversation_mapping:
                new_ticket_id = conversation_mapping[old_conv_id]
                
                tickets_cursor.execute('''
                    INSERT INTO messages (ticket_id, sender, text, telegram_message_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    new_ticket_id,
                    msg_dict['sender'],
                    msg_dict['text'],
                    msg_dict.get('telegram_message_id'),
                    msg_dict['created_at']
                ))
                migrated_messages += 1
        
        print(f"✅ Мигрировано сообщений: {migrated_messages}")
        
        # Мигрируем базу знаний
        print("🧠 Миграция базы знаний...")
        try:
            old_cursor.execute("SELECT * FROM knowledge_entries")
            knowledge_entries = old_cursor.fetchall()
            
            old_cursor.execute("PRAGMA table_info(knowledge_entries)")
            know_columns_info = old_cursor.fetchall()
            know_col_names = [col[1] for col in know_columns_info]
            
            for entry_data in knowledge_entries:
                entry_dict = dict(zip(know_col_names, entry_data))
                
                knowledge_cursor.execute('''
                    INSERT INTO knowledge_entries (question, answer, embedding, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (
                    entry_dict['question'],
                    entry_dict['answer'],
                    entry_dict['embedding'],
                    entry_dict['created_at']
                ))
            
            print(f"✅ Мигрировано записей базы знаний: {len(knowledge_entries)}")
            
        except sqlite3.OperationalError:
            print("ℹ️  Таблица knowledge_entries не найдена в старой БД")
        
        # Подтверждаем транзакции
        tickets_conn.commit()
        knowledge_conn.commit()
        
        print("🎉 Миграция завершена успешно!")
        print(f"   • База заявок: {tickets_db_path}")
        print(f"   • База знаний: {knowledge_db_path}")
        print(f"   • Старая БД сохранена: {old_db_path}")
        
    except Exception as e:
        print(f"❌ Ошибка при миграции: {e}")
        tickets_conn.rollback()
        knowledge_conn.rollback()
        
    finally:
        old_conn.close()
        tickets_conn.close()
        knowledge_conn.close()


if __name__ == "__main__":
    asyncio.run(migrate_to_tickets())