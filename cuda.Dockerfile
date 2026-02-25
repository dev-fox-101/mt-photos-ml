# ghcr.io/immich-app/immich-machine-learning:v2.5.6-cuda  - "architecture": "amd64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:6bfdbe69ffcd0543f8f8fc45993d9b1559aa3c92078c44b41ece67132d16d9d3

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
