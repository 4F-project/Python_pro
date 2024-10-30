import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# R-CNN 모델 로드 
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # 평가 모드로 전환

# 이미지 전처리 함수
def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # 이미지를 Tensor로 변환
    ])

# 탐지 결과 시각화 함수
def plot_image_with_boxes(image, boxes, labels, scores):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    # 각 박스, 레이블, 점수를 시각화
    for box, label, score in zip(boxes, labels, scores):
        box = box.detach().cpu().numpy()  # NumPy 배열로 변환
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                 linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(box[0], box[1], f'Class {label.item()}: {score.item():.2f}',
                 color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.show()

# 이미지 폴더 경로 설정
image_folder = 'C:\\Users\\hi\\dev\\proj1\\screenshot'  # 이미지가 저장된 폴더 경로
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 신뢰도 임계값 설정
confidence_threshold = 0.2  # 0.2 이상인 객체만 선택

# 각 이미지 처리
for file in image_files:
    image_path = os.path.join(image_folder, file)

    # 이미지 로드 및 전처리
    image = Image.open(image_path)
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

    # R-CNN으로 객체 탐지
    with torch.no_grad():  # 그래디언트 계산 비활성화
        outputs = model(image_tensor)

    # 탐지 결과에서 박스, 레이블, 신뢰도 추출
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    # 신뢰도 기준으로 선택된 객체 필터링
    selected_indices = scores > confidence_threshold  # 신뢰도가 임계값 이상인 인덱스
    selected_boxes = boxes[selected_indices]
    selected_labels = labels[selected_indices]
    selected_scores = scores[selected_indices]

    # 선택된 객체가 없으면 넘어가기
    if selected_boxes.numel() == 0:  # selected_boxes가 비어있는 경우 체크
        print(f"No objects detected in {file}.")
        continue

    # 탐지된 객체 시각화
    plot_image_with_boxes(image, selected_boxes, selected_labels, selected_scores)
