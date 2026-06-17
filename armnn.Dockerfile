# ghcr.io/immich-app/immich-machine-learning:v2.7.5-armnn  - "architecture": "arm64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:3f5be135aabdfe28678b240b778b4efdd80dc4a281412427771d8ed9777f79e7

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
EXPOSE 3003
