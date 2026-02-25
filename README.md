# MT Photos 智能识别+人脸识别API

基于 immich-machine-learning:v2.5.6 镜像构建，为MT Photos 提供CLIP识别、文本识别、人脸识别的API；

修改了main.py部分代码，重新打包镜像覆盖/usr/src/immich_ml/main.py文件；


- CLIP识别模型使用 `XLM-Roberta-Base-ViT-B-32__laion5b_s13b_b90k`
  - 这个模型相较于官方镜像中的 cn-clip 模型，解决了不认识颜色的问题
- OCR识别模型使用 `PP-OCRv5_mobile`
  - 与官方镜像效果差不多 
- 人脸识别模型使用 `buffalo_l`
  - 与官方镜像效果一样


镜像原始代码地址：

https://github.com/immich-app/immich/tree/v2.5.6/machine-learning

---
本项目GitHub代码仓库地址：

https://github.com/dev-fox-101/mt-photos-ml


## docker镜像说明

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

---

## 添加api使用

返回MT Photos的系统设置中，在智能识别API 和 人脸识别API配置中；

**接口地址** 填写 **http://[nas的IP]:8060**

**API_AUTH_KEY** 填写 **mt_photos_ai_extra**

---

### 清除旧数据+重新识别

--- 

#### CLIP

如果之前已经使用`mtphotos/mt-photos-ai`处理了 **CLIP识别**；

由于模型的变化，需要清除旧数据后重新识别，操作步骤为： 

- 1、关闭CLIP识别，并且确认当前没有在执行的后台任务
- 2、在系统维护工具中，执行 **【CLIP识别】- 清空识别结果，然后重新识别所有照片** 任务
- 3、等待后台任务中 **重置CLIP识别状态中** 任务完成后，再开启CLIP识别
- 4、等待 CLIP识别识别完成

--- 

#### 文本识别 

文本识别 **不需要**清空旧数据+重新识别；

--- 

#### 人脸识别 

如果之前使用的是 `devfox101/mt-photos-insightface-unofficial` 或者 `devfox101/mt-photos-ai` 镜像，并且人脸识别模型是默认的 buffalo_l 模型，那么**不需要**清空旧数据+重新识别；

如果之前使用的不是insightface相关的镜像， 可以按这个步骤清空数据重新识别：

- 1、先关闭人脸识别开关，确认没有正在运行的人脸识别任务；

- 2、在工具=>系统维护工具 中执行 【人物相册】- 重建全部数据。建议在清空识别结果前，对数据库进行一次备份；

- 3、等待后台任务中数据清除任务完成之后，再切换内置识别模型或人脸识别API

- 4、填写API配置，注意：当换了人脸识别的API后，请检查人脸置信度和匹配差异值阈值是否为安装教程中推荐的值

- 5、最后打开人脸识别开关，查看后台任务信息


