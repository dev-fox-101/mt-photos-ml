import asyncio
import gc
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from io import BytesIO
from typing import Any, AsyncGenerator, Callable, Iterator
from zipfile import BadZipFile

import cv2
import numpy as np
import orjson
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, ORJSONResponse, PlainTextResponse
from numpy.typing import NDArray
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf, NoSuchFile
from PIL import Image
from pydantic import BaseModel, ValidationError
from starlette.formparsers import MultiPartParser

from immich_ml.models import get_model_deps
from immich_ml.models.base import InferenceModel
from immich_ml.models.transforms import decode_cv2, decode_pil, pil_to_cv2, serialize_np_array

from .config import PreloadModelData, log, settings
from .models.cache import ModelCache
from .schemas import (
    InferenceEntries,
    InferenceEntry,
    InferenceResponse,
    ModelFormat,
    ModelIdentity,
    ModelTask,
    ModelType,
    PipelineRequest,
    T,
)

# API 认证密钥
api_auth_key = os.getenv("API_AUTH_KEY", "mt_photos_ai_extra")

# 模型名称配置
ocr_model_name = os.getenv("OCR_MODEL_NAME", "PP-OCRv5_mobile")
clip_visual_model_name = os.getenv("CLIP_VISUAL_MODEL_NAME", "XLM-Roberta-Base-ViT-B-32__laion5b_s13b_b90k")
clip_textual_model_name = os.getenv("CLIP_TEXTUAL_MODEL_NAME", "XLM-Roberta-Base-ViT-B-32__laion5b_s13b_b90k")
face_detection_model_name = os.getenv("FACE_DETECTION_MODEL_NAME", "buffalo_l")
face_recognition_model_name = os.getenv("FACE_RECOGNITION_MODEL_NAME", "buffalo_l")
face_min_score = float(os.getenv("FACE_MIN_SCORE", "0.65"))


class ClipTxtRequest(BaseModel):
    text: str

MultiPartParser.spool_max_size = 2**26  # spools to disk if payload is 64 MiB or larger

model_cache = ModelCache(revalidate=settings.model_ttl > 0)
thread_pool: ThreadPoolExecutor | None = None
lock = threading.Lock()
active_requests = 0
last_called: float | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    global thread_pool
    log.info(
        (
            "Created in-memory cache with unloading "
            f"{f'after {settings.model_ttl}s of inactivity' if settings.model_ttl > 0 else 'disabled'}."
        )
    )

    try:
        if settings.request_threads > 0:
            # asyncio is a huge bottleneck for performance, so we use a thread pool to run blocking code
            thread_pool = ThreadPoolExecutor(settings.request_threads) if settings.request_threads > 0 else None
            log.info(f"Initialized request thread pool with {settings.request_threads} threads.")
        if settings.model_ttl > 0 and settings.model_ttl_poll_s > 0:
            asyncio.ensure_future(idle_shutdown_task())
        if settings.preload is not None:
            await preload_models(settings.preload)
        yield
    finally:
        log.handlers.clear()
        for model in model_cache.cache._cache.values():
            del model
        if thread_pool is not None:
            thread_pool.shutdown()
        gc.collect()


async def preload_models(preload: PreloadModelData) -> None:
    log.info(f"Preloading models: clip:{preload.clip} facial_recognition:{preload.facial_recognition}")

    async def load_models(model_string: str, model_type: ModelType, model_task: ModelTask) -> None:
        for model_name in model_string.split(","):
            model_name = model_name.strip()
            model = await model_cache.get(model_name, model_type, model_task)
            await load(model)

    if preload.clip.textual is not None:
        await load_models(preload.clip.textual, ModelType.TEXTUAL, ModelTask.SEARCH)

    if preload.clip.visual is not None:
        await load_models(preload.clip.visual, ModelType.VISUAL, ModelTask.SEARCH)

    if preload.facial_recognition.detection is not None:
        await load_models(
            preload.facial_recognition.detection,
            ModelType.DETECTION,
            ModelTask.FACIAL_RECOGNITION,
        )

    if preload.facial_recognition.recognition is not None:
        await load_models(
            preload.facial_recognition.recognition,
            ModelType.RECOGNITION,
            ModelTask.FACIAL_RECOGNITION,
        )

    if preload.ocr.detection is not None:
        await load_models(
            preload.ocr.detection,
            ModelType.DETECTION,
            ModelTask.OCR,
        )

    if preload.ocr.recognition is not None:
        await load_models(
            preload.ocr.recognition,
            ModelType.RECOGNITION,
            ModelTask.OCR,
        )

    if preload.clip_fallback is not None:
        log.warning(
            "Deprecated env variable: 'MACHINE_LEARNING_PRELOAD__CLIP'. "
            "Use 'MACHINE_LEARNING_PRELOAD__CLIP__TEXTUAL' and "
            "'MACHINE_LEARNING_PRELOAD__CLIP__VISUAL' instead."
        )

    if preload.facial_recognition_fallback is not None:
        log.warning(
            "Deprecated env variable: 'MACHINE_LEARNING_PRELOAD__FACIAL_RECOGNITION'. "
            "Use 'MACHINE_LEARNING_PRELOAD__FACIAL_RECOGNITION__DETECTION' and "
            "'MACHINE_LEARNING_PRELOAD__FACIAL_RECOGNITION__RECOGNITION' instead."
        )


def update_state() -> Iterator[None]:
    global active_requests, last_called
    active_requests += 1
    last_called = time.time()
    try:
        yield
    finally:
        active_requests -= 1


def get_entries(entries: str = Form()) -> InferenceEntries:
    try:
        request: PipelineRequest = orjson.loads(entries)
        without_deps: list[InferenceEntry] = []
        with_deps: list[InferenceEntry] = []
        for task, types in request.items():
            for type, entry in types.items():
                parsed: InferenceEntry = {
                    "name": entry["modelName"],
                    "task": task,
                    "type": type,
                    "options": entry.get("options", {}),
                }
                dep = get_model_deps(parsed["name"], type, task)
                (with_deps if dep else without_deps).append(parsed)
        return without_deps, with_deps
    except (orjson.JSONDecodeError, ValidationError, KeyError, AttributeError) as e:
        log.error(f"Invalid request format: {e}")
        raise HTTPException(422, "Invalid request format.")


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping() -> PlainTextResponse:
    return PlainTextResponse("pong")


async def verify_header(api_key: str = Header(...)) -> str:
    """验证 API 密钥"""
    if api_key != api_auth_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


@app.get("/", response_class=HTMLResponse)
async def top_info() -> str:
    """服务首页信息"""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>MT Photos AI Server</title>
    <style>p{text-align: center;}</style>
</head>
<body>
<p style="font-weight: 600;">MT Photos智能识别服务</p>
<p>服务状态： 运行中</p>
<p>使用方法： <a href="https://mtmt.tech/docs/advanced/ocr_api">https://mtmt.tech/docs/advanced/ocr_api</a></p>
</body>
</html>"""
    return html_content


@app.post("/check")
async def check_req(api_key: str = Depends(verify_header)) -> dict[str, Any]:
    """检查服务状态"""
    return {
        'result': 'pass',
        "title": "mt-photos-ai服务",
        "help": "https://mtmt.tech/docs/advanced/ocr_api",
        "detector_backend": "insightface",
        "recognition_model": face_recognition_model_name,
        "facial_min_score": face_min_score,
        "facial_max_distance": 0.5,
        "env_use_dml": False
    }


@app.post("/restart")
async def restart(api_key: str = Depends(verify_header)) -> dict[str, str]:
    """重启服务接口（预留）"""
    return {'result': 'pass'}


@app.post("/restart_v2")
async def restart_v2(api_key: str = Depends(verify_header)) -> dict[str, str]:
    """重启服务接口 V2（预留）"""
    return {'result': 'pass'}


def to_fixed(num: float) -> str:
    """将数字格式化为保留2位小数的字符串"""
    return str(round(num, 2))


def trans_ocr_result(result: Any) -> dict[str, Any]:
    """转换 OCR 结果格式以兼容 server.py"""
    texts = []
    scores = []
    boxes = []
    if result is None:
        return {'texts': texts, 'scores': scores, 'boxes': boxes}
    for res_i in result:
        dt_box = res_i[0]
        box = {
            'x': to_fixed(dt_box[0][0]),
            'y': to_fixed(dt_box[0][1]),
            'width': to_fixed(dt_box[1][0] - dt_box[0][0]),
            'height': to_fixed(dt_box[2][1] - dt_box[0][1])
        }
        boxes.append(box)
        texts.append(res_i[1])
        scores.append(f"{res_i[2]:.2f}")
    return {'texts': texts, 'scores': scores, 'boxes': boxes}


@app.post("/ocr")
async def ocr_endpoint(
        file: bytes = File(..., description="Image file"),
        api_key: str = Depends(verify_header)
) -> dict[str, Any]:
    """OCR 文字识别接口"""
    image_bytes = file
    try:
        # 检查图片尺寸
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {'result': [], 'msg': 'Invalid image format'}
        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        # 使用 predict 逻辑进行 OCR
        pil_img = Image.open(BytesIO(image_bytes))
        pil_img.load()
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # 构建 OCR pipeline 请求
        ocr_request: PipelineRequest = {
            ModelTask.OCR: {
                ModelType.DETECTION: {"modelName": ocr_model_name, "options": {}},
                ModelType.RECOGNITION: {"modelName": ocr_model_name, "options": {}}
            }
        }

        entries = parse_pipeline_request(ocr_request)
        response = await run_inference(pil_img, entries)

        # 转换结果为 server.py 格式
        if ModelTask.OCR in response:
            ocr_output = response[ModelTask.OCR]
            if isinstance(ocr_output, dict):
                result_list = []
                texts = ocr_output.get("text", [])
                box_scores = ocr_output.get("boxScore", np.array([]))
                text_scores = ocr_output.get("textScore", np.array([]))
                boxes = ocr_output.get("box", np.array([]))

                # 将扁平化的 box 数组重塑为 4x2 格式
                if len(boxes) > 0:
                    boxes_reshaped = boxes.reshape(-1, 4, 2)
                    for i, text in enumerate(texts):
                        if i < len(boxes_reshaped):
                            box = boxes_reshaped[i]
                            score = float(text_scores[i]) if i < len(text_scores) else 0.0
                            box_score = float(box_scores[i]) if i < len(box_scores) else 0.0
                            # 转换为 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 格式
                            box_formatted = [[float(box[j][0] * width), float(box[j][1] * height)] for j in range(4)]
                            result_list.append([box_formatted, text, score])

                # 按 Y 坐标排序（从上到下）
                result_list.sort(key=lambda x: x[0][0][1])
                formatted_result = trans_ocr_result(result_list)
                return {'result': formatted_result}

        return {'result': {'texts': [], 'scores': [], 'boxes': []}}
    except Exception as e:
        log.error(f"OCR error: {e}")
        return {'result': [], 'msg': str(e)}


def parse_pipeline_request(request: PipelineRequest) -> InferenceEntries:
    """解析 pipeline 请求为 inference entries"""
    without_deps: list[InferenceEntry] = []
    with_deps: list[InferenceEntry] = []
    for task, types in request.items():
        for type, entry in types.items():
            parsed: InferenceEntry = {
                "name": entry["modelName"],
                "task": task,
                "type": type,
                "options": entry.get("options", {}),
            }
            dep = get_model_deps(parsed["name"], type, task)
            (with_deps if dep else without_deps).append(parsed)
    return without_deps, with_deps


@app.post("/clip/img")
async def clip_process_image(
        file: bytes = File(..., description="Image file"),
        api_key: str = Depends(verify_header)
) -> dict[str, Any]:
    """CLIP 图像编码接口"""
    image_bytes = file
    try:
        # 检查图片尺寸
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {'result': [], 'msg': 'Invalid image format'}

        # 解码 PIL 图像
        pil_img = await run(lambda: decode_pil(image_bytes))

        # 构建 CLIP visual pipeline 请求
        clip_request: PipelineRequest = {
            ModelTask.SEARCH: {
                ModelType.VISUAL: {"modelName": clip_visual_model_name, "options": {}}
            }
        }

        entries = parse_pipeline_request(clip_request)
        response = await run_inference(pil_img, entries)

        # 解析 embedding 结果
        if ModelTask.SEARCH in response:
            embedding_str = response[ModelTask.SEARCH]
            if isinstance(embedding_str, str):
                embedding = orjson.loads(embedding_str)
                # 格式化为 server.py 的返回格式
                return {'result': ["{:.16f}".format(vec) for vec in embedding]}

        return {'result': []}
    except Exception as e:
        log.error(f"CLIP image error: {e}")
        return {'result': [], 'msg': str(e)}


@app.post("/clip/txt")
async def clip_process_txt(
        request: ClipTxtRequest,
        api_key: str = Depends(verify_header)
) -> dict[str, Any]:
    """CLIP 文本编码接口"""
    try:
        text = request.text

        # 构建 CLIP textual pipeline 请求
        clip_request: PipelineRequest = {
            ModelTask.SEARCH: {
                ModelType.TEXTUAL: {"modelName": clip_textual_model_name, "options": {}}
            }
        }

        entries = parse_pipeline_request(clip_request)
        response = await run_inference(text, entries)

        # 解析 embedding 结果
        if ModelTask.SEARCH in response:
            embedding_str = response[ModelTask.SEARCH]
            if isinstance(embedding_str, str):
                embedding = orjson.loads(embedding_str)
                # 格式化为 server.py 的返回格式
                return {'result': ["{:.16f}".format(vec) for vec in embedding]}

        return {'result': []}
    except Exception as e:
        log.error(f"CLIP text error: {e}")
        return {'result': [], 'msg': str(e)}


@app.post("/represent")
async def represent_endpoint(
        file: bytes = File(..., description="Image file"),
        content_type: str = Header(default="image/jpeg", alias="content-type"),
        api_key: str = Depends(verify_header)
) -> dict[str, Any]:
    """人脸识别接口"""
    image_bytes = file
    try:
        img = None
        # 处理 GIF 文件
        if content_type == 'image/gif':
            with Image.open(BytesIO(image_bytes)) as pil_img:
                if pil_img.is_animated:
                    pil_img.seek(0)
                frame = pil_img.convert('RGB')
                np_arr = np.array(frame)
                img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)

        if img is None:
            # 使用 OpenCV 解码其他图片类型
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            err = f"The uploaded file is not a valid image format or is corrupted."
            return {'result': [], 'msg': str(err)}

        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        data = {
            "detector_backend": "insightface",
            "recognition_model": face_recognition_model_name
        }

        # 构建人脸识别 pipeline 请求
        face_request: PipelineRequest = {
            ModelTask.FACIAL_RECOGNITION: {
                ModelType.DETECTION: {"modelName": face_detection_model_name, "options": {"minScore": face_min_score}},
                ModelType.RECOGNITION: {"modelName": face_recognition_model_name, "options": {}}
            }
        }

        entries = parse_pipeline_request(face_request)

        # 将 OpenCV 图像转换为 PIL 用于推理
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        response = await run_inference(pil_img, entries)

        # 转换结果格式
        if ModelTask.FACIAL_RECOGNITION in response:
            faces = response[ModelTask.FACIAL_RECOGNITION]
            if isinstance(faces, list):
                embedding_objs = []
                for face in faces:
                    embedding_str = face.get("embedding", "[]")
                    embedding = orjson.loads(embedding_str)
                    bbox = face.get("boundingBox", {})
                    score = face.get("score", 0.0)

                    x1, y1 = bbox.get("x1", 0), bbox.get("y1", 0)
                    x2, y2 = bbox.get("x2", 0), bbox.get("y2", 0)
                    resp_obj = {
                        "embedding": embedding,
                        "facial_area": {
                            "x": int(x1),
                            "y": int(y1),
                            "w": int(x2 - x1),
                            "h": int(y2 - y1)
                        },
                        "face_confidence": float(score)
                    }
                    embedding_objs.append(resp_obj)

                data["result"] = embedding_objs
                return data

        data["result"] = []
        return data
    except Exception as e:
        if 'set enforce_detection' in str(e):
            return {'result': []}
        log.error(f"Face recognition error: {e}")
        return {'result': [], 'msg': str(e)}


@app.post("/predict", dependencies=[Depends(update_state)])
async def predict(
        entries: InferenceEntries = Depends(get_entries),
        image: bytes | None = File(default=None),
        text: str | None = Form(default=None),
) -> Any:
    if image is not None:
        inputs: Image.Image | str = await run(lambda: decode_pil(image))
    elif text is not None:
        inputs = text
    else:
        raise HTTPException(400, "Either image or text must be provided")
    response = await run_inference(inputs, entries)
    return ORJSONResponse(response)


async def run_inference(payload: Image.Image | str, entries: InferenceEntries) -> InferenceResponse:
    outputs: dict[ModelIdentity, Any] = {}
    response: InferenceResponse = {}

    async def _run_inference(entry: InferenceEntry) -> None:
        model = await model_cache.get(
            entry["name"], entry["type"], entry["task"], ttl=settings.model_ttl, **entry["options"]
        )
        inputs = [payload]
        for dep in model.depends:
            try:
                inputs.append(outputs[dep])
            except KeyError:
                message = f"Task {entry['task']} of type {entry['type']} depends on output of {dep}"
                raise HTTPException(400, message)
        model = await load(model)
        output = await run(model.predict, *inputs, **entry["options"])
        outputs[model.identity] = output
        response[entry["task"]] = output

    without_deps, with_deps = entries
    await asyncio.gather(*[_run_inference(entry) for entry in without_deps])
    if with_deps:
        await asyncio.gather(*[_run_inference(entry) for entry in with_deps])
    if isinstance(payload, Image.Image):
        response["imageHeight"], response["imageWidth"] = payload.height, payload.width

    return response


async def run(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    if thread_pool is None:
        return func(*args, **kwargs)
    partial_func = partial(func, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(thread_pool, partial_func)


async def load(model: InferenceModel) -> InferenceModel:
    if model.loaded:
        return model

    def _load(model: InferenceModel) -> InferenceModel:
        if model.load_attempts > 1:
            raise HTTPException(500, f"Failed to load model '{model.model_name}'")
        with lock:
            try:
                model.load()
            except FileNotFoundError as e:
                if model.model_format == ModelFormat.ONNX:
                    raise e
                log.warning(
                    f"{model.model_format.upper()} is available, but model '{model.model_name}' does not support it.",
                    exc_info=e,
                )
                model.model_format = ModelFormat.ONNX
                model.load()
        return model

    try:
        return await run(_load, model)
    except (OSError, InvalidProtobuf, BadZipFile, NoSuchFile):
        log.warning(f"Failed to load {model.model_type.replace('_', ' ')} model '{model.model_name}'. Clearing cache.")
        model.clear_cache()
        return await run(_load, model)


async def idle_shutdown_task() -> None:
    while True:
        if (
                last_called is not None
                and not active_requests
                and not lock.locked()
                and time.time() - last_called > settings.model_ttl
        ):
            log.info("Shutting down due to inactivity.")
            os.kill(os.getpid(), signal.SIGINT)
            break
        await asyncio.sleep(settings.model_ttl_poll_s)
