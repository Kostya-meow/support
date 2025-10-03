"""
Миграция: добавление полей first_response_at и summary в таблицу tickets
"""
import sqlite3
from pathlib import Path

# Путь к БД
db_path = Path(__file__).parent / "tickets.db"

def migrate():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Добавляем поле first_response_at
        cursor.execute("""
            ALTER TABLE tickets 
            ADD COLUMN first_response_at DATETIME
        """)
        print("✓ Добавлено поле first_response_at")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("  Поле first_response_at уже существует")
        else:
            raise
    
    try:
        # Добавляем поле summary
        cursor.execute("""
            ALTER TABLE tickets 
            ADD COLUMN summary TEXT
        """)
        print("✓ Добавлено поле summary")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("  Поле summary уже существует")
        else:
            raise
    
    conn.commit()
    conn.close()
    print("\n✅ Миграция успешно выполнена!")

if __name__ == "__main__":
    migrate()
