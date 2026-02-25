# ghcr.io/immich-app/immich-machine-learning:v2.5.6  - "architecture": "amd64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:eb05df5c977c2620f37e1a3e60962eafe77d8b222a770eda96ae38c5014bf0ba

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
