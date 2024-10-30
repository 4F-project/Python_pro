from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

# 로컬 이미지 로드
IMAGE_FILE = "wine.jpg"  # 실제 파일 경로로 변경
image = Image.open(IMAGE_FILE)

# 프로세서와 모델 로드
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

# 이미지 처리
inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# 출력 후 처리
target_sizes = torch.tensor([image.size[::-1]])  # 높이, 너비
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

# 결과 출력
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"탐지된 객체: {model.config.id2label[label.item()]} (신뢰도: {round(score.item(), 3)}) 위치: {box}"
    )

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_image_with_boxes(image, results):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(box[0], box[1], f'{model.config.id2label[label.item()]}: {round(score.item(), 2)}',
                 color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

# 결과 시각화
plot_image_with_boxes(image, results)

