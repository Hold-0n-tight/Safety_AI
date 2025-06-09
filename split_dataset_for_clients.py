import os
import shutil
import random
from glob import glob
from pathlib import Path

# 원본 데이터 경로
SOURCE_DIR = "/Users/hold_0n_tight/Downloads/안공지능/Dataset/NCT-CRC-HE-100K"
TARGET_BASE = "/Users/hold_0n_tight/federated-learning-docker"  # 여기를 네 프로젝트 루트로 맞춰줘
NUM_CLIENTS = 5
SEED = 42

random.seed(SEED)

# 클래스별로 나누기
classes = os.listdir(SOURCE_DIR)

for class_name in classes:
    class_dir = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    # 클래스 내 모든 이미지 가져오기
    images = glob(os.path.join(class_dir, "*.tif"))
    random.shuffle(images)

    # 이미지를 client 수로 나누기
    split_size = len(images) // NUM_CLIENTS

    for i in range(NUM_CLIENTS):
        client_id = f"client{i+1}"
        start = i * split_size
        end = (i+1) * split_size if i < NUM_CLIENTS - 1 else len(images)

        client_class_dir = os.path.join(TARGET_BASE, client_id, "data", class_name)
        os.makedirs(client_class_dir, exist_ok=True)

        for img_path in images[start:end]:
            shutil.copy(img_path, client_class_dir)

print("✅ 클라이언트별 데이터셋 분리 완료!")
