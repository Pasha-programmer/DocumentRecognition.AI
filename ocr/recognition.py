import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sys
import io
import base64
import logging

logger = logging.getLogger(__name__)

def get_model(num_classes: int) -> nn.Module:
    """
    Создаёт архитектуру модели, совместимую с сохранённой.
    Используется ResNet18 с заменой последнего полносвязного слоя.
    """
    model = models.resnet18(weights=None)  # веса загрузим позже из чекпоинта
    # Заменяем классификатор: Dropout + Linear
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model


def load_model(checkpoint_path: str, device: torch.device):
    """
    Загружает модель и мета-информацию из файла чекпоинта.
    Возвращает (model, idx_to_label, transform, device)
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Файл модели не найден: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Извлекаем параметры
    num_classes = checkpoint['num_classes']
    img_size = checkpoint['img_size']
    mean = checkpoint['mean']
    std = checkpoint['std']
    idx_to_label = checkpoint['idx_to_label']  # словарь {index: label}
    label_to_idx = checkpoint['label_to_idx']  # может пригодиться, но не обязательно

    # Создаём модель и загружаем веса
    model = get_model(num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Трансформации для входного изображения (должны совпадать с валидационными)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return model, idx_to_label, transform


def predict_image(model, image_data: bytes, transform, device, idx_to_label, top_k: int = 1):
    """
    Классифицирует одно изображение.
    Возвращает список (класс, вероятность) для top_k предсказаний.
    """
    try:
        # io.BytesIO создает файлоподобный объект в памяти
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки изображения из Blob: {e}")
    
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Инференс
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    # Получаем top-k вероятностей и индексы
    top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    top_probs = top_probs.cpu().numpy().flatten()
    top_indices = top_indices.cpu().numpy().flatten()

    results = []
    for i, idx in enumerate(top_indices):
        label = idx_to_label[idx]

        results.append((label, top_probs[i]))

    return results

def start_recognition(image_blob: str, model: str, top_k: int, useCpu: bool):
    """Начать распознавание."""

    # Декодируем Base64, если пришла строка
    if isinstance(image_blob, str):
        # Если строка содержит заголовок 'data:image/png;base64,...', его нужно отрезать
        if ',' in image_blob:
            image_blob = image_blob.split(',')[1]
        image_bytes = base64.b64decode(image_blob)
    else:
        image_bytes = image_blob

    logger.info(f"Параметры распознавания bytes.count: {image_bytes.count} model: {model} top_k: {top_k} useCpu: {useCpu}")

    # Определяем устройство
    if useCpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем модель
    try:
        model, idx_to_label, transform = load_model(model, device)
        logger.info(f"Модель {model} успешно загружена.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}", file=sys.stderr)
        raise e

    # Распознаём изображение
    try:
        predictions = predict_image(model, image_bytes, transform, device, idx_to_label, top_k)
    except Exception as e:
        logger.error(f"Ошибка при распознавании: {e}", file=sys.stderr)
        raise e

    return predictions
    