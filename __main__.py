import logging

from database.sqlite import init_db
from rabbit_mq.consumer import RabbitMQConsumer

logger = logging.getLogger(__name__)

def main():
    """Основная функция"""

    init_db()

    consumer = RabbitMQConsumer()
    
    try:
        consumer.connect()
        consumer.consume()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        consumer.stop()

if __name__ == "__main__":
    main()