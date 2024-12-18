import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory


# Загрузка предобученной YOLO модели
model = YOLO('yolov8n.pt')  # Модель YOLO от Ultralytics

def visualize_predictions_with_yolo(images, model, class_names=None, max_images=1000, save_dir=None):
    """
    Визуализация предсказаний YOLO с рамками и текстом только для кошек и собак.
    Сохранение изображений с результатами, если указан save_dir.
    """
    num_images = min(images.shape[0], max_images)

    # Создаём директорию для сохранения изображений, если она указана
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Индексы классов для кошек и собак (COCO dataset)
    cat_idx = 15  # Класс "cat"
    dog_idx = 16  # Класс "dog"
    target_classes = {cat_idx, dog_idx}

    for i in range(num_images):
        # Преобразуем тензор в numpy-формат
        img = images[i].numpy().astype("uint8")
        if img.shape[-1] == 4:  # Если есть альфа-канал, удаляем его
            img = img[..., :3]

        # YOLO требует формат BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Детекция объектов с использованием YOLO
        results = model.predict(source=img_bgr, save=False, imgsz=640, conf=0.25)

        # Копируем изображение для отрисовки рамок
        img_with_boxes = img_bgr.copy()
        for result in results:
            for box in result.boxes:
                # Получаем индекс класса объекта
                class_idx = int(box.cls[0])  # Индекс класса

                # Фильтруем только кошек и собак
                if class_idx in target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf[0]  # Уверенность
                    label = class_names[class_idx] if class_names and class_idx in class_names else str(class_idx)

                    # Рисуем рамку и текст
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Зелёные рамки
                    cv2.putText(
                        img_with_boxes, f"{label} {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                    )

        # Конвертируем обратно в RGB для отображения в Matplotlib
        img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

        # Визуализируем результат
        plt.figure(figsize=(10, 5))
        plt.imshow(img_rgb)
        plt.axis("off")

        # Сохраняем изображение или отображаем
        if save_dir:
            output_path = os.path.join(save_dir, f"result_{i + 1}.png")
            plt.savefig(output_path, bbox_inches="tight")
        else:
            plt.show()

        plt.close()  # Освобождаем память, закрывая текущую фигуру


# Датасет валидации
val_dataset = image_dataset_from_directory(
    'Verify',
    image_size=(640, 640),  # Размер изображений для YOLO
    batch_size=128
)

# Названия классов
class_names = model.names  # Список классов модели YOLO

# Директория для сохранения результатов
save_directory = "Results"

# Применяем YOLO к данным
for images, labels in val_dataset.take(1):  # Берём одну партию данных
    visualize_predictions_with_yolo(images, model, class_names, max_images=1000, save_dir=save_directory)
