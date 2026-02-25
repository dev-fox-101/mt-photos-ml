# ghcr.io/immich-app/immich-machine-learning:v2.5.6  - "architecture": "arm64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:4aeb46b8993c90ece5d3577bcb9d6545923dd77afc3585507057fd5b2d1fd9f0

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
