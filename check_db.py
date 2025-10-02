import sqlite3

# Проверяем базу знаний
print("=== KNOWLEDGE DB ===")
conn = sqlite3.connect('knowledge.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print('Tables:', cursor.fetchall())
cursor.execute('SELECT COUNT(*) FROM knowledge_entries;')
print('Entries:', cursor.fetchone()[0])
conn.close()

# Проверяем базу заявок
print("\n=== TICKETS DB ===")
conn = sqlite3.connect('tickets.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print('Tables:', cursor.fetchall())
cursor.execute('SELECT COUNT(*) FROM tickets;')
print('Tickets:', cursor.fetchone()[0])
cursor.execute('SELECT COUNT(*) FROM messages;')
print('Messages:', cursor.fetchone()[0])
conn.close()