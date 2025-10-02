import sqlite3

# Проверяем содержимое базы данных заявок
conn = sqlite3.connect('tickets.db')
cursor = conn.cursor()

cursor.execute('SELECT * FROM tickets;')
print('=== TICKETS ===')
for row in cursor.fetchall():
    print(row)

cursor.execute('SELECT * FROM messages ORDER BY created_at;')
print('\n=== MESSAGES ===')
for row in cursor.fetchall():
    print(row)

conn.close()