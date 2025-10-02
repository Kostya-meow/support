import sqlite3

def migrate_add_is_system_field():
    """Добавляет поле is_system в таблицу messages"""
    
    conn = sqlite3.connect('tickets.db')
    cursor = conn.cursor()
    
    try:
        # Проверяем, есть ли уже поле is_system
        cursor.execute("PRAGMA table_info(messages)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'is_system' not in columns:
            print("Добавляем поле is_system в таблицу messages...")
            cursor.execute("ALTER TABLE messages ADD COLUMN is_system BOOLEAN DEFAULT 0 NOT NULL")
            print("Поле is_system добавлено успешно!")
        else:
            print("Поле is_system уже существует")
            
        conn.commit()
        
    except Exception as e:
        print(f"Ошибка миграции: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_add_is_system_field()