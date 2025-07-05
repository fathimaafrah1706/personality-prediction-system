import sqlite3

# Connect to the database
conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

# Fetch all rows
cursor.execute("SELECT * FROM predictions")
rows = cursor.fetchall()

# Print each row
for row in rows:
    print(row)

conn.close()
