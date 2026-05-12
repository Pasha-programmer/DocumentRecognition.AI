import pika
import json
import logging
from typing import Optional
from src.config import Config

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RabbitMQPublisher:
    def __init__(self):

        self.config = Config()

        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None
        
    def connect(self):
        """Установка соединения с RabbitMQ"""
        try:
            # Создаем credentials
            credentials = pika.PlainCredentials(self.config.RABBITMQ_USER, self.config.RABBITMQ_PASSWORD)
            
            # Параметры соединения
            parameters = pika.ConnectionParameters(
                host=self.config.RABBITMQ_HOST,
                port=self.config.RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            
            # Устанавливаем соединение
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # Объявляем очередь (создаем если не существует)
            self.channel.queue_declare(
                queue=self.config.RABBITMQ_QUEUE2,
                durable=True,  # Очередь сохранится после перезапуска
                exclusive=False,
                auto_delete=False
            )
            
            logger.info(f"Успешно подключились к RabbitMQ. Очередь: {self.config.RABBITMQ_QUEUE2}")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к RabbitMQ: {e}")
            raise
    
    def publish_message(
        self, 
        message: dict | list | str, 
        content_type: str = 'application/json',
        persistent: bool = True,
        priority: Optional[int] = None
    ) -> bool:
        """
        Публикация сообщения в очередь
        
        Args:
            message: Сообщение для отправки (будет преобразовано в JSON если не строка)
            content_type: Тип содержимого
            persistent: Сделать сообщение персистентным (сохранить на диск)
            priority: Приоритет сообщения (1-10, где 10 - наивысший)
            
        Returns:
            bool: Успешность публикации
        """
        if not self.channel or self.channel.is_closed:
            logger.warning("Канал не открыт, пытаемся переподключиться")
            if not self.connect():
                return False
        
        try:
            # Преобразуем сообщение в строку
            if isinstance(message, (dict, list)):
                body = json.dumps(message, ensure_ascii=False)
            elif isinstance(message, str):
                body = message
            else:
                body = str(message)
            
            # Создаем свойства сообщения
            properties = {
                'content_type': content_type,
                'delivery_mode': 2 if persistent else 1,  # 2 = persistent, 1 = non-persistent
            }
            
            if priority is not None:
                properties['priority'] = min(10, max(1, priority))  # Ограничиваем от 1 до 10
            
            # Публикуем сообщение
            self.channel.basic_publish(
                exchange="",
                routing_key=self.config.RABBITMQ_QUEUE2,
                body=body.encode('utf-8'),
                properties=pika.BasicProperties(**properties)
            )
            
            logger.debug(f"Сообщение отправлено: {body[:100]}...")
            return True
            
        except pika.exceptions.AMQPError as e:
            logger.error(f"Ошибка при публикации сообщения: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при публикации: {e}")
            return False
    
    def close(self):
        """Закрытие соединения с RabbitMQ"""
        try:
            if self.channel and not self.channel.is_closed:
                self.channel.close()
            if self.connection and self.connection.is_open:
                self.connection.close()
            logger.info("Соединение с RabbitMQ закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения: {e}")
    
    def __enter__(self):
        """Контекстный менеджер для автоматического подключения"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Закрытие соединения при выходе из контекста"""
        self.close()
