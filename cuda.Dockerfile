# ghcr.io/immich-app/immich-machine-learning:v2.7.5-cuda  - "architecture": "amd64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:5987079143add2ad60e8c2155d9575aba96d4cdffa60c0381c8a1ef74dc00e11

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
EXPOSE 3003
