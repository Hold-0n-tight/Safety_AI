import os
import shutil
from glob import glob

# 테스트 데이터 원본
TEST_SOURCE_DIR = "/Users/hold_0n_tight/Downloads/안공지능/Dataset/CRC-VAL-HE-7K"
TARGET_DIR = "/Users/hold_0n_tight/federated-learning-docker/server/test/data"

classes = os.listdir(TEST_SOURCE_DIR)

for class_name in classes:
    class_path = os.path.join(TEST_SOURCE_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    target_class_dir = os.path.join(TARGET_DIR, class_name)
    os.makedirs(target_class_dir, exist_ok=True)

    images = glob(os.path.join(class_path, "*.tif"))
    for img_path in images:
        shutil.copy(img_path, target_class_dir)

print("✅ 테스트 데이터 서버로 복사 완료!")
