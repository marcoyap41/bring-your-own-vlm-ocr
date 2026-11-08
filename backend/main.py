import ast
import base64
import io
import json
import os
import re
import time

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from openai import APIError, OpenAI, RateLimitError
from PIL import Image, ImageOps

load_dotenv()

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "google/gemma-4-31b-it"
MAX_IMAGE_SIDE = 2048
MIN_IMAGE_SIDE = 32
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

FULL_PAGE_PROMPT = (
    "Please output only the text content from the image without any additional "
    "descriptions or formatting."
)
SPOTTING_PROMPT = (
    "Detect all text lines in the image with line-level granularity. "
    "Output ONLY a valid JSON array of objects, where each object has keys "
    "'text_content' and 'bbox_2d' ([x1, y1, x2, y2] coordinates normalized to 0-1000, "
    "where x is horizontal from 0 left to 1000 right, and y is vertical from 0 top to 1000 bottom)."
)

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["*"],
        }
    },
)


class HttpError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@app.errorhandler(HttpError)
def handle_http_error(err: HttpError):
    return jsonify({"detail": err.detail}), err.status_code


def get_base_url() -> str:
    return (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or DEFAULT_BASE_URL
    ).strip()


def get_api_key() -> str:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or ""
    ).strip()


def get_model_name() -> str:
    return (
        os.getenv("MODEL_NAME")
        or os.getenv("OPENROUTER_MODEL")
        or DEFAULT_MODEL_NAME
    ).strip()


def get_client() -> OpenAI:
    api_key = get_api_key()
    base_url = get_base_url()
    if not api_key:
        raise HttpError(500, "OPENAI_API_KEY is not set in .env")
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def prepare_image(file_storage) -> tuple[str, int, int]:
    if not file_storage:
        raise HttpError(400, "No image file provided")

    if file_storage.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
        raise HttpError(400, "Only JPG, PNG, or WEBP images are supported")

    data = file_storage.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HttpError(400, "Image is too large (max 15 MB)")

    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
    except Exception:
        raise HttpError(400, "Could not read the uploaded image")

    if image.mode != "RGB":
        image = image.convert("RGB")

    if min(image.size) < MIN_IMAGE_SIDE:
        raise HttpError(400, f"Image must be at least {MIN_IMAGE_SIDE}px on each side")

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded, image.width, image.height


def build_messages(base64_image: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a precise OCR assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ]


def create_raw_completion(client: OpenAI, messages: list[dict], stream: bool):
    max_tokens = 2048
    kwargs = {
        "model": get_model_name(),
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "stream": stream,
    }
    if stream:
        kwargs["stream_options"] = {"include_usage": True}

    for attempt in range(3):
        try:
            return client.chat.completions.with_raw_response.create(**kwargs)
        except RateLimitError:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))


def extract_rate_limits(headers) -> dict:
    def to_int(key: str):
        val = headers.get(key)
        try:
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "limit_tokens": to_int("x-ratelimit-limit-tokens"),
        "remaining_tokens": to_int("x-ratelimit-remaining-tokens"),
        "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
        "limit_requests": to_int("x-ratelimit-limit-requests"),
        "remaining_requests": to_int("x-ratelimit-remaining-requests"),
        "reset_requests": headers.get("x-ratelimit-reset-requests"),
    }


def extract_usage(usage_obj, elapsed_sec: float | None = None) -> dict | None:
    if not usage_obj:
        return None
    total_time = getattr(usage_obj, "total_time", None)
    if total_time is None and elapsed_sec is not None:
        total_time = elapsed_sec
    return {
        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage_obj, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage_obj, "total_tokens", 0) or 0,
        "total_time": round(float(total_time or 0.0), 3),
    }


def strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_json(raw: str):
    text = strip_think(raw)

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()

    brackets = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
    if brackets:
        text = brackets.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            objects = []
            for match in re.finditer(r"\{[^{}]*\}", text):
                try:
                    obj = json.loads(match.group(0))
                    if isinstance(obj, dict) and ("bbox_2d" in obj or "bbox" in obj):
                        objects.append(obj)
                except Exception:
                    continue
            return objects if objects else None


def normalize_boxes(parsed, width: int, height: int) -> list[dict]:
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
    if not isinstance(parsed, list):
        return []

    items = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        coords = item.get("bbox_2d") or item.get("bbox")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue
        try:
            coords = [float(c) for c in coords]
        except (TypeError, ValueError):
            continue
        items.append((str(item.get("text_content", item.get("text", ""))), coords))

    if not items:
        return []

    half = len(items) // 2
    if (
        len(items) % 2 == 0
        and half > 0
        and [t for t, _ in items[:half]] == [t for t, _ in items[half:]]
    ):
        items = items[:half]

    max_value = max(max(coords) for _, coords in items)
    if max_value <= 1.0:
        sx, sy = 1.0, 1.0
    elif max_value <= 1000.0:
        sx, sy = 1000.0, 1000.0
    else:
        sx, sy = float(width), float(height)

    boxes = []
    for text, (x1, y1, x2, y2) in items:
        nx1, nx2 = sorted((x1 / sx, x2 / sx))
        ny1, ny2 = sorted((y1 / sy, y2 / sy))
        boxes.append(
            {
                "text": text,
                "box": [
                    round(min(max(nx1, 0.0), 1.0), 5),
                    round(min(max(ny1, 0.0), 1.0), 5),
                    round(min(max(nx2, 0.0), 1.0), 5),
                    round(min(max(ny2, 0.0), 1.0), 5),
                ],
            }
        )
    return boxes


def stream_events(completion, rate_limit: dict, start_time: float):
    yield json.dumps({"type": "rate_limit", "data": rate_limit}) + "\n"

    buffer = ""
    in_think = False
    open_tag, close_tag = "<think>", "</think>"
    usage_data = None

    for chunk in completion:
        if getattr(chunk, "usage", None):
            usage_data = extract_usage(chunk.usage, time.perf_counter() - start_time)

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        buffer += getattr(delta, "content", None) or ""

        while buffer:
            if in_think:
                end = buffer.find(close_tag)
                if end == -1:
                    buffer = buffer[-(len(close_tag) - 1) :]
                    break
                buffer = buffer[end + len(close_tag) :].lstrip()
                in_think = False
            else:
                start = buffer.find(open_tag)
                if start != -1:
                    if start > 0:
                        yield json.dumps({"type": "token", "text": buffer[:start]}) + "\n"
                    buffer = buffer[start + len(open_tag) :]
                    in_think = True
                    continue
                safe = len(buffer)
                for i in range(1, len(open_tag)):
                    if buffer.endswith(open_tag[:i]):
                        safe = len(buffer) - i
                        break
                if safe > 0:
                    yield json.dumps({"type": "token", "text": buffer[:safe]}) + "\n"
                buffer = buffer[safe:]
                break

    if buffer and not in_think:
        yield json.dumps({"type": "token", "text": buffer}) + "\n"

    if usage_data:
        yield json.dumps({"type": "usage", "data": usage_data}) + "\n"


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "model": get_model_name(),
            "base_url": get_base_url(),
            "api_key_configured": bool(get_api_key()),
        }
    )


@app.post("/api/ocr/full")
def ocr_full():
    client = get_client()
    file_storage = request.files.get("file")
    base64_image, _, _ = prepare_image(file_storage)

    start_time = time.perf_counter()
    try:
        raw_response = create_raw_completion(
            client, build_messages(base64_image, FULL_PAGE_PROMPT), stream=True
        )
        rate_limit = extract_rate_limits(raw_response.headers)
        completion = raw_response.parse()
    except APIError as e:
        raise HttpError(502, f"API error: {e.message}")
    except Exception as e:
        raise HttpError(502, f"API error: {e}")

    def generator():
        try:
            for event in stream_events(completion, rate_limit, start_time):
                yield event
        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return Response(stream_with_context(generator()), mimetype="application/x-ndjson")


@app.post("/api/ocr/spotting")
def ocr_spotting():
    client = get_client()
    file_storage = request.files.get("file")
    base64_image, width, height = prepare_image(file_storage)

    start_time = time.perf_counter()
    try:
        raw_response = create_raw_completion(
            client, build_messages(base64_image, SPOTTING_PROMPT), stream=False
        )
        rate_limit = extract_rate_limits(raw_response.headers)
        completion = raw_response.parse()
    except APIError as e:
        raise HttpError(502, f"API error: {e.message}")
    except Exception as e:
        raise HttpError(502, f"API error: {e}")

    elapsed = time.perf_counter() - start_time
    raw = completion.choices[0].message.content or ""
    boxes = normalize_boxes(parse_json(raw), width, height)
    usage = extract_usage(getattr(completion, "usage", None), elapsed)

    return jsonify(
        {
            "boxes": boxes,
            "raw": strip_think(raw),
            "width": width,
            "height": height,
            "usage": usage,
            "rate_limit": rate_limit,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
