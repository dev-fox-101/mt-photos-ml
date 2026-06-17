# ghcr.io/immich-app/immich-machine-learning:v2.7.5-rocm  - "architecture": "amd64"
FROM ghcr.io/immich-app/immich-machine-learning@sha256:95c21706ccfa8cfa54ffb4733eacecf4be0dc31d3209a8aa82408a4bd51bafaa

# 覆盖镜像内部的main.py，添加 /ocr 、/clip/img 、/clip/txt 、/represent 这些接口
COPY ./main.py /usr/src/immich_ml/main.py
EXPOSE 3003
