import pika
import json
import logging
from typing import Optional
from database.sqlite import executeSqlCommand
from rabbit_mq.publisher import RabbitMQPublisher
from src.config import Config
from ocr.recognition import start_recognition
from ocr.tune import tune_model 

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RabbitMQConsumer:

    aiModelNamesMap = {
        "v1.1": "glagolitic_model_full_v1_1",
        "v2.0": "glagolitic_model_full_v2_0",
        "v2.1": "glagolitic_model_full_v2_1",
        "v2.2": "glagolitic_model_full_v2_2",
        "v3.0": "glagolitic_model_full_v3_0",
        "v4.0": "glagolitic_model_full_v4_0",
    }

    aiTunedModelNamesMap = {
        "v1.1": "glagolitic_model_full_v1_1_tuned",
        "v2.0": "glagolitic_model_full_v2_0_tuned",
        "v2.1": "glagolitic_model_full_v2_1_tuned",
        "v2.2": "glagolitic_model_full_v2_2_tuned",
        "v3.0": "glagolitic_model_full_v3_0_tuned",
        "v4.0": "glagolitic_model_full_v4_0_tuned",
    }

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
            self.channel.queue_declare(queue=self.config.RABBITMQ_QUEUE_RECOGNITION_REQUEST, durable=True)
            
            logger.info(f"Успешно подключились к RabbitMQ. Очередь: {self.config.RABBITMQ_QUEUE_RECOGNITION_REQUEST}")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к RabbitMQ: {e}")
            raise
    
    def process_recognition_request_message(self, ch, method, properties, body):
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

            modelNames = self.getAiModelFileNames(message['AiModelType'])

            response_payload_list = []

            for modelName in modelNames:
                predictions = start_recognition(message['Blob'], "./aiModels/" + modelName + ".pth", 3, False)

                for i, (label, prob) in enumerate(predictions):
                    float_prob = float(prob)
                    modelTypeKey = next((k for k, v in self.aiModelNamesMap.items() if v == modelName), None)

                    response_payload_list.append({
                        "DocumentId": message['DocumentId'],
                        "Label": label,
                        "Probability": float(prob),
                        "ModelType": modelTypeKey,
                        "RecognitionType": "Auto"
                    })

                    executeSqlCommand(f'''
                        INSERT INTO DocumentPrediction
                        (DocumentId, ModelType, Label, Prob)
                        VALUES({message['DocumentId']}, '{modelTypeKey}', '{label}', {float_prob});
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

    def process_tune_request_message(self, ch, method, properties, body):
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

            modelNames = self.getAiModelFileNames(message['AiModelType'])

            for modelName in modelNames:
                tune_model(message['RootDir'], 
                           message['NewDataFileName'], 
                           "./aiModels/" + modelName + ".pth",
                           "./aiModels/" + modelName + "_tuned.pth")
            
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
                    queue=self.config.RABBITMQ_QUEUE_RECOGNITION_REQUEST,
                    on_message_callback=self.process_recognition_request_message
                )

                self.channel.basic_consume(
                    queue=self.config.RABBITMQ_QUEUE_TUNE_REQUEST,
                    on_message_callback=self.process_tune_request_message
                )
                
                logger.info(f"Ожидание сообщений в очереди {self.config.RABBITMQ_QUEUE_RECOGNITION_REQUEST}. Для выхода нажмите CTRL+C")
                
                # Запускаем цикл получения сообщений
                self.channel.start_consuming()
                
            except KeyboardInterrupt:
                logger.info("Получен сигнал остановки")
                self.stop()
            except Exception as e:
                logger.error(f"Ошибка в процессе потребления: {e}")
                self.stop()

    def getAiModelFileNames(self, aiModelType: str):
        modelNames = []

        if (aiModelType != "All"):
            modelTypeName = self.aiModelNamesMap.get(aiModelType)

            if (modelTypeName == None):
                raise Exception("Не удалось определить тип модели распознавания")
            
            modelNames.append(modelTypeName)
            return modelNames

        for modelTypeName in self.aiModelNamesMap.values():
            modelNames.append(modelTypeName)

        return modelNames
    
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