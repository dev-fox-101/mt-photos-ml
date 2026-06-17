# ghcr.io/immich-app/immich-machine-learning:v2.7.5-rknn  - "architecture": "arm64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:c1831fc9fc408613e63c1cb52ffb021eaa251054de7da6ea33893a2013c2af58

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
EXPOSE 3003
