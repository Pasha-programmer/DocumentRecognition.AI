import sqlite3
import os

from src.config import Config

def executeSqlCommand(sqlCommand: str):

    config = Config()

    # Устанавливаем соединение с базой данных
    connection = sqlite3.connect(config.DB_CONNECTION_STRING)
    cursor = connection.cursor()

    # Создаем таблицу Users
    cursor.execute(sqlCommand)

    # Сохраняем изменения и закрываем соединение
    connection.commit()
    connection.close()

def init_db():
    config = Config()

    # Убедимся, что директория существует (на случай, если volume пуст)
    os.makedirs(os.path.dirname(config.DB_CONNECTION_STRING), exist_ok=True)

    executeSqlCommand('''
        CREATE TABLE IF NOT EXISTS DocumentPrediction (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            DocumentId TEXT NOT NULL,
            ModelType TEXT NOT NULL,
            Label TEXT NOT NULL,
            Prob INTEGER NOT NULL
        )
    ''')