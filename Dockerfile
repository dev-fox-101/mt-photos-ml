# ghcr.io/immich-app/immich-machine-learning:v2.7.5 - "architecture": "amd64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:c7a8bd9cc982024a55da94c235122449ff8fa91347b6e99f902f31e0349fc623

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
EXPOSE 3003
