import torch
import clip
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# CLIP 모델 및 프로세서 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 폴더 경로 설정
image_folder = 'C:\\Users\\hi\\dev\\proj1\\screenshot'  # 이미지가 저장된 폴더 경로
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 여러 이미지를 로드하고 전처리
images = []
for file in image_files:
    image_path = os.path.join(image_folder, file)
    try:
        img = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        images.append(img)
    except Exception as e:
        print(f"Error loading image {file}: {e}")

# 이미지 특징 추출
with torch.no_grad():
    image_features = torch.cat([clip_model.encode_image(img) for img in images], dim=0)

# 유사도 행렬 계산 (코사인 유사도)
similarity_matrix = (image_features @ image_features.T).cpu().numpy()

# 클러스터링 (KMeans)
num_clusters = 10  # 클러스터 수를 10으로 증가
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(image_features.cpu().numpy())
clusters = kmeans.labels_

# 결과 시각화
def plot_clusters(image_files, clusters):
    cluster_dict = {}
    for idx, cluster in enumerate(clusters):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(image_files[idx])
    
    for cluster, files in cluster_dict.items():
        plt.figure(figsize=(15, 5))
        plt.title(f"Cluster {cluster}")
        for i, file in enumerate(files):
            plt.subplot(1, len(files), i + 1)
            img = Image.open(os.path.join(image_folder, file))
            plt.imshow(img)
            plt.axis('off')
            plt.title(file)
        plt.show()

# 클러스터 결과 시각화
plot_clusters(image_files, clusters)

# 유사도 결과 출력
for i in range(len(image_files)):
    for j in range(i + 1, len(image_files)):
        print(f"Cosine Similarity between {image_files[i]} and {image_files[j]}: {similarity_matrix[i][j]:.4f}")
