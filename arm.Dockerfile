# ghcr.io/immich-app/immich-machine-learning:v2.7.5  - "architecture": "arm64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:d7057237a74a9219d890b0e00cb59434327720e65f991118c5e5c4f05092b694

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
EXPOSE 3003
