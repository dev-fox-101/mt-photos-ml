# ghcr.io/immich-app/immich-machine-learning:v2.5.6-openvino  - "architecture": "amd64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:54b7f03e9e02b2f8e7f600628c1619e66cae502ad0d3569859f408f584917ad8

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
EXPOSE 3003
