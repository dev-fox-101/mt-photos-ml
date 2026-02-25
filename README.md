# MT Photos 智能识别+人脸识别API

基于 immich-machine-learning:v2.5.6 镜像构建，针对MT Photos 的CLIP识别、文本识别、人脸识别的API请求和返回数据格式，在原始镜像中修改了main.py部分代码；


镜像原始代码地址：

https://github.com/immich-app/immich/tree/v2.5.6/machine-learning


gitHub代码仓库地址：
https://github.com/dev-fox-101/mt-photos-ml


## 镜像说明

DockerHub镜像仓库地址：
https://hub.docker.com/r/devfox101/mt-photos-ml

镜像Tags说明：

- `devfox101/mt-photos-ml:v2.5.6`：x86架构的CPU处理识别任务，所有Intel、AMD 处理器都可以运行这个镜像
- `devfox101/mt-photos-ml:v2.5.6-arm`：arm64架构的CPU处理识别任务， arm架构的nas、以及m系列芯片的mac可以运行这个镜像
- `devfox101/mt-photos-ml:v2.5.6-cuda`：Nvidia显卡使用GPU处理识别任务，需要Nvidia显卡
- `devfox101/mt-photos-ml:v2.5.6-openvino`：Intel CPU使用OpenVINO调用核显或者独立显卡处理识别任务


## 支持配置的环境变量

- **API_AUTH_KEY**：API认证密钥，用于保护API接口，防止被恶意访问， 默认值为： `mt_photos_ai_extra`
- **OCR_MODEL_NAME**：OCR模型名称，用于指定使用哪个OCR模型进行文字识别，默认值为： `PP-OCRv5_mobile`
- **CLIP_VISUAL_MODEL_NAME**：CLIP模型名称，用于指定使用哪个CLIP模型进行图片识别，默认值为： `XLM-Roberta-Base-ViT-B-32__laion5b_s13b_b90k`
- **CLIP_TEXTUAL_MODEL_NAME**：CLIP模型名称，用于指定使用哪个CLIP模型进行文本处理，默认值为： `XLM-Roberta-Base-ViT-B-32__laion5b_s13b_b90k`
- **FACE_DETECTION_MODEL_NAME**：人脸检测模型名称，用于指定使用哪个模型进行人脸检测，默认值为： `buffalo_l`
- **FACE_RECOGNITION_MODEL_NAME**：人脸特征提取模型名称，用于指定使用哪个模型进行人脸特征提取，默认值为： `buffalo_l`
- **FACE_MIN_SCORE**：人脸检测最小置信度，默认值为： `0.65`


### CLIP模型选择

`XLM-Roberta-Base-ViT-B-32__laion5b_s13b_b90k` 和 `XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k`
对于中文特有的一些元素搜索效果相教与nllb模型更好，比如：春节、鞭炮之类; 因此使用了XLM-Roberta-Base-ViT-B-32__laion5b_s13b_b90k 作为默认值；

查看支持的CLIP模型：https://docs.immich.app/features/searching/#clip-models

#### 容器环境变量中配置CLIP模型

在创建容器时，增加这2个环境变量： `-e CLIP_VISUAL_MODEL_NAME=XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k -e CLIP_TEXTUAL_MODEL_NAME=XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k` 

可以指定CLIP模型使用XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k


#### CLIP模型返回的向量长度

默认CLIP模型返回的向量长度为512，比如使用的模型返回的向量长度不是512，需要在 智能识别API配置那边通过 自定义CLIP向量长度 来修改；

比如以下CLIP模型返回的向量长度不是512：

- 'XLM-Roberta-Large-ViT-H-14__frozen_laion5b_s13b_b90k': { dimSize: 1024 }
- 'nllb-clip-base-siglip__mrl': { dimSize: 768 }
- 'nllb-clip-base-siglip__v1': { dimSize: 768 }
- 'nllb-clip-large-siglip__mrl': { dimSize: 1152 }
- 'nllb-clip-large-siglip__v1': { dimSize: 1152 }


## 下载镜像、创建docker容器

注意：需要有目录映射 /cache，它是用于存储各种识别模型的文件

x86 CPU：
```bash
docker pull devfox101/mt-photos-ml:v2.5.6

docker run -i -p 8060:3003 -e API_AUTH_KEY=mt_photos_ai_extra -v /mnt/mt-photos/ml-cache:/cache --name mt-photos-ml --restart="unless-stopped" devfox101/mt-photos-ml:v2.5.6
```

arm64 CPU：
```bash
docker pull devfox101/mt-photos-ml:v2.5.6-arm

docker run -i -p 8060:3003 -e API_AUTH_KEY=mt_photos_ai_extra -v /mnt/mt-photos/ml-cache:/cache --name mt-photos-ml --restart="unless-stopped" devfox101/mt-photos-ml:v2.5.6-arm
```


cuda加速 ：
```bash
docker pull devfox101/mt-photos-ml:v2.5.6-cuda

docker run --gpus all -i -p 8060:3003 -e API_AUTH_KEY=mt_photos_ai_extra -v /mnt/mt-photos/ml-cache:/cache --name mt-photos-ml --restart="unless-stopped" devfox101/mt-photos-ml:v2.5.6-cuda
```

openvino加速 ：
```bash
docker pull devfox101/mt-photos-ml:v2.5.6-openvino

docker run -i -p 8060:3003 -e API_AUTH_KEY=mt_photos_ai_extra --device /dev/dri:/dev/dri -v /mnt/mt-photos/ml-cache:/cache --name mt-photos-ml --restart="unless-stopped" devfox101/mt-photos-ml:v2.5.6-openvino
```
