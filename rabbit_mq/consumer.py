import pika
import json
import logging
from typing import Optional
from database.sqlite import executeSqlCommand
from rabbit_mq.publisher import RabbitMQPublisher
from src.config import Config
from ocr.recognition import start_recognition

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RabbitMQConsumer:
    def __init__(self):

        self.config = Config()

        self.publisher = RabbitMQPublisher()

        self.should_stop = False

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
            
            # Объявляем очередь (на случай, если она еще не создана)
            self.channel.queue_declare(queue=self.config.RABBITMQ_QUEUE, durable=True)
            
            logger.info(f"Успешно подключились к RabbitMQ. Очередь: {self.config.RABBITMQ_QUEUE}")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к RabbitMQ: {e}")
            raise
    
    def process_message(self, ch, method, properties, body):
        """Обработка полученного сообщения"""
        try:
            # Пытаемся распарсить JSON
            try:
                message = json.loads(body)
                logger.info(f"Получено сообщение (JSON)")
            except json.JSONDecodeError:
                # Если не JSON, обрабатываем как строку
                message = body.decode('utf-8')
                logger.info(f"Получено сообщение (строка)")

            modelNames = {
                "GlagoliticModelFullV1": "glagolitic_model_full",
                "GlagoliticModelFullV2": "glagolitic_model_full_v2",
                "GlagoliticModelFullV3": "glagolitic_model_full_v3"
            }

            modelName = modelNames.get(message['Model'])

            if (modelName == None):
                raise Exception("Не удалось определить тип модели распознавания")

            predictions = start_recognition(message['Blob'], "./" + modelName + ".pth", 3, True)

            response_payload_list = []

            for i, (label, prob) in enumerate(predictions):
                float_prob = float(prob)

                response_payload_list.append({
                    "DocumentId": message['DocumentId'],
                    "Label": label,
                    "Probability": float(prob),
                    "ModelType": message['Model']
                })

                executeSqlCommand(f'''
                    INSERT INTO DocumentPrediction
                    (DocumentId, ModelType, Label, Prob)
                    VALUES({message['DocumentId']}, '{message['Model']}', '{label}', {float_prob});
                ''')

            response_payload = json.dumps(response_payload_list)

            logger.info(response_payload)

            try:
                self.publisher.connect()
                self.publisher.publish_message(response_payload)
            except Exception as e:
                logger.error(f"Критическая ошибка: {e}")
                self.publisher.close()
                raise e

            # Подтверждаем обработку сообщения
            ch.basic_ack(delivery_tag=method.delivery_tag)

            logger.info(f"Сообщение обработано и подтверждено")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке сообщения: {e}")
            # Отклоняем сообщение и не возвращаем в очередь
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    
    def consume(self):
        """Запуск consumer'а"""
        while not self.should_stop:
            try:
                if not self.connection or self.connection.is_closed:
                    self.connect()
                
                # Настраиваем получение только одного сообщения за раз
                self.channel.basic_qos(prefetch_count=1)
                
                # Подписываемся на очередь
                self.channel.basic_consume(
                    queue=self.config.RABBITMQ_QUEUE,
                    on_message_callback=self.process_message
                )
                
                logger.info(f"Ожидание сообщений в очереди {self.config.RABBITMQ_QUEUE}. Для выхода нажмите CTRL+C")
                
                # Запускаем цикл получения сообщений
                self.channel.start_consuming()
                
            except KeyboardInterrupt:
                logger.info("Получен сигнал остановки")
                self.stop()
            except Exception as e:
                logger.error(f"Ошибка в процессе потребления: {e}")
                self.stop()
    
    def stop(self):
        """Остановка consumer'а и закрытие соединения"""
        try:
            if self.channel and self.channel.is_open:
                self.channel.stop_consuming()
            
            if self.connection and self.connection.is_open:
                self.connection.close()
                logger.info("Соединение с RabbitMQ закрыто")
        except Exception as e:
            logger.error(f"Ошибка при закрытии соединения: {e}")