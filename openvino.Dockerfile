# ghcr.io/immich-app/immich-machine-learning:v2.7.5-openvino  - "architecture": "amd64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:619b43773d4f050d7504a236061034c06ef653f28693813ebccb1ab5d1e77512

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
EXPOSE 3003
