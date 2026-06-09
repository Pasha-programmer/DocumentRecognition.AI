import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Dataset, Трансформации и DataLoader
class GlagoliticDataset(Dataset):
    def __init__(self, samples, img_size=224, transform=None):
        self.samples = samples
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')
        if self.transform:
            image = self.transform(image)
        return image, label

# Собираем базовую архитектуру
def rebuild_model(num_classes):
    # Загружаем базовый resnet18 (веса подтянутся из вашего чекпоинта, поэтому тут можно без IMAGENET1K_V1)
    model = models.resnet18() 
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, num_classes)
    )
    return model

# Подготовка новых данных (используя старый маппинг классов)
def parse_new_dataset(csv_path, root_dir, label_to_idx):
    df = pd.read_csv(csv_path, header=None, names=["paths", "label"], delimiter=';')
    samples = []
    for _, row in df.iterrows():
        label = row["label"]
        # Защита: проверяем, знает ли модель такой класс из старого набора
        if label not in label_to_idx:
            print(f"Предупреждение: Пропуск неизвестной метки класса '{label}'")
            continue
            
        path_items = str(row["paths"]).split('|')
        for item in path_items:
            item = item.strip()
            full_path = os.path.join(root_dir, item)
            if os.path.isdir(full_path):
                for fname in os.listdir(full_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(full_path, fname)
                        samples.append((img_path, label_to_idx[label]))
            elif os.path.isfile(full_path):
                samples.append((full_path, label_to_idx[label]))
    return samples

# Дообучить модель
# root_dir - путь к папке
# new_data_file_name - Новый CSV-файл с данными
# checkpoint_file_name - Путь к сохраненной модели
# new_checkpoint_file_name - Имя для нового чекпоинта
def tune_model(root_dir, new_data_file_name, checkpoint_file_path, new_checkpoint_file_path):
    # root_dir = "/content/drive/MyDrive/SlavonicRecognition/train_v2_2"
    # new_data_file_name = "/content/drive/MyDrive/SlavonicRecognition/NewMap.csv"  # Новый CSV-файл с данными
    # checkpoint_file_name = "glagolitic_model_full.pth"  # Путь к вашей сохраненной модели
    # new_checkpoint_file_name = "glagolitic_model_tuned.pth" # Имя для нового чекпоинта

    BATCH_SIZE = 16
    EPOCHS = 20  # Обычно для тонкой настройки на новых данных нужно меньше эпох
    LEARNING_RATE = 1e-5  # Очень маленький LR, чтобы не "сломать" старые веса
    WEIGHT_DECAY = 1e-4

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используется устройство: {DEVICE}")

    # ---------------------------
    # 2. Загрузка чекпоинта и восстановление архитектуры
    # ---------------------------
    logger.info("Загрузка чекпоинта...")
    checkpoint = torch.load(checkpoint_file_path, map_location=DEVICE)

    # Извлекаем метаданные
    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = checkpoint['idx_to_label']
    num_classes = checkpoint['num_classes']
    img_size = checkpoint['img_size']
    mean = checkpoint['mean']
    std = checkpoint['std']

    model = rebuild_model(num_classes)
    # Загружаем веса
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    logger.info(f"Модель успешно загружена. Количество классов: {num_classes}")

    new_samples = parse_new_dataset(new_data_file_name, root_dir, label_to_idx)
    logger.info(f"Новых изображений для дообучения загружено: {len(new_samples)}")

    # Применяем аугментации (такие же, как были при обучении)
    fine_tune_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = GlagoliticDataset(new_samples, img_size=img_size, transform=fine_tune_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # ---------------------------
    # 5. Настройка оптимизатора и обучение
    # ---------------------------
    criterion = nn.CrossEntropyLoss()

    # Размораживаем всю модель для деликатного дообучения (Fine-Tuning)
    for param in model.parameters():
        param.requires_grad = True

    # Используем крайне низкий lr, чтобы не разрушить паттерны в нижних слоях
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model.train()
    logger.info("\n=== Старт дообучения на новых данных ===")

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        logger.info(f"Эпоха [{epoch+1}/{EPOCHS}] | Потери (Loss): {epoch_loss:.4f} | Точность (Acc): {epoch_acc:.4f}")

    # Сохранение обновленной модели
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'num_classes': num_classes,
        'img_size': img_size,
        'mean': mean,
        'std': std
    }, new_checkpoint_file_path)

    logger.info(f"\nОбновленная модель успешно сохранена в '{new_checkpoint_file_path}'")