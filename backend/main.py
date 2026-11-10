import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
import re
import sqlite3
import threading
import time
import urllib.request

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from PIL import Image, ImageFilter, ImageOps

load_dotenv()

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "google/gemma-4-31b-it"
MAX_IMAGE_SIDE = 2048
MIN_IMAGE_SIDE = 32
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
TARGET_STRIP_HEIGHT = 650
MAX_STRIPS = 4

DEMO_MAX_REQUESTS = 5
DEMO_MAX_INPUT_TOKENS = 2000
DB_PATH = os.path.join(os.path.dirname(__file__), "demo_quota.db")
_DB_LOCK = threading.Lock()


def is_demo_mode_enabled() -> bool:
    """
    Demo Mode is only enforced when explicitly enabled via ENABLE_DEMO_MODE=true.
    When False (default for personal/local self-hosting), server-level .env config
    is treated as a full unlimited personal instance without 5-request limits.
    """
    raw = os.getenv("ENABLE_DEMO_MODE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

FULL_PAGE_PROMPT = (
    "Please output only the text content from the image without any additional "
    "descriptions or formatting."
)
GEMMA_SPOTTING_PROMPT = (
    "Detect every single individual horizontal line of text in the image from top to bottom. "
    "If text wraps across 2 or more lines, output a separate bounding box for EACH individual line of text "
    "(do NOT merge multiple lines into one box, and do NOT skip the second or lower lines). "
    "Return ONLY a valid JSON array of objects, where each object has keys "
    "'box_2d' ([ymin, xmin, ymax, xmax] normalized 0-1000) and 'label' (the exact text string of that single line)."
)
GENERIC_SPOTTING_PROMPT = (
    "Detect every single individual horizontal line of text in the image from top to bottom. "
    "If text wraps across 2 or more lines, output a separate bounding box for EACH individual line of text "
    "(do NOT merge multiple lines into one box, and do NOT skip the second or lower lines). "
    "Output ONLY a valid JSON array of objects with keys 'text_content' and "
    "'bbox_2d' ([xmin, ymin, xmax, ymax] normalized 0-1000)."
)


def init_db() -> None:
    with _DB_LOCK:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS demo_usage (
                    identifier TEXT PRIMARY KEY,
                    requests_used INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()


init_db()


def is_ymin_xmin_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(k in lower for k in ("gemma", "gemini", "google/", "paligemma"))


def get_spotting_prompt(model_name: str) -> str:
    if is_ymin_xmin_model(model_name):
        return GEMMA_SPOTTING_PROMPT
    return GENERIC_SPOTTING_PROMPT


FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

app = Flask(__name__, static_folder=FRONTEND_DIST if os.path.exists(FRONTEND_DIST) else None)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "*",
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
    ).strip().rstrip("/")


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


def get_request_config() -> tuple[str, str, str, bool]:
    """
    Resolve (base_url, api_key, model_name, is_custom_api) for the active request.
    Uses client-supplied BYOK headers exclusively in-memory if X-Api-Key is present;
    otherwise falls back to the server's .env demo configuration.
    """
    custom_key = (request.headers.get("X-Api-Key") or "").strip()
    if custom_key:
        custom_url = (request.headers.get("X-Base-Url") or "").strip().rstrip("/") or get_base_url()
        custom_model = (request.headers.get("X-Model-Name") or "").strip() or get_model_name()
        return custom_url, custom_key, custom_model, True
    return get_base_url(), get_api_key(), get_model_name(), False


def get_client_identifiers() -> tuple[str, str | None]:
    forwarded = (
        request.headers.get("CF-Connecting-IP")
        or request.headers.get("X-Forwarded-For")
        or request.remote_addr
        or "unknown"
    )
    client_ip = forwarded.split(",")[0].strip() or "unknown"
    ip_key = f"ip:{client_ip}"

    raw_cid = (request.headers.get("X-Client-Id") or "").strip()
    cid_key = f"cid:{raw_cid[:64]}" if raw_cid else None
    return ip_key, cid_key


def get_demo_quota_status() -> dict | None:
    if not is_demo_mode_enabled():
        return None

    ip_key, cid_key = get_client_identifiers()
    keys = [ip_key] + ([cid_key] if cid_key else [])
    placeholders = ",".join("?" for _ in keys)

    with _DB_LOCK:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                f"SELECT requests_used FROM demo_usage WHERE identifier IN ({placeholders})",
                keys,
            ).fetchall()

    used = max((r[0] for r in rows), default=0)
    return {
        "limit_requests": DEMO_MAX_REQUESTS,
        "used_requests": min(DEMO_MAX_REQUESTS, used),
        "remaining_requests": max(0, DEMO_MAX_REQUESTS - used),
        "max_input_tokens": DEMO_MAX_INPUT_TOKENS,
    }


def consume_demo_quota() -> dict | None:
    if not is_demo_mode_enabled():
        return None

    ip_key, cid_key = get_client_identifiers()
    keys = [ip_key] + ([cid_key] if cid_key else [])
    placeholders = ",".join("?" for _ in keys)
    now = time.time()

    with _DB_LOCK:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                f"SELECT requests_used FROM demo_usage WHERE identifier IN ({placeholders})",
                keys,
            ).fetchall()
            used = max((r[0] for r in rows), default=0)
            if used >= DEMO_MAX_REQUESTS:
                raise HttpError(
                    429,
                    f"Free Demo API limit reached ({DEMO_MAX_REQUESTS}/{DEMO_MAX_REQUESTS} requests used). "
                    "Please click 'Set your own API' to configure your personal API key.",
                )
            next_used = used + 1
            for k in keys:
                conn.execute(
                    """
                    INSERT INTO demo_usage (identifier, requests_used, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(identifier) DO UPDATE SET
                        requests_used = excluded.requests_used,
                        updated_at = excluded.updated_at
                    """,
                    (k, next_used, now),
                )
            conn.commit()

    return {
        "limit_requests": DEMO_MAX_REQUESTS,
        "used_requests": next_used,
        "remaining_requests": max(0, DEMO_MAX_REQUESTS - next_used),
        "max_input_tokens": DEMO_MAX_INPUT_TOKENS,
    }


def fetch_openrouter_key_quota(base_url: str, api_key: str) -> dict | None:
    """
    Fetch per-key cost usage and credit limit from GET {base_url}/key
    (supported by OpenRouter API keys).
    """
    if not api_key or "openrouter.ai" not in base_url.lower():
        return None
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/key",
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "bring-your-own-vlm-ocr",
            },
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        data = payload.get("data") or {}
        eff = data.get("effective_budget") or {}

        usage_usd = data.get("usage")
        if usage_usd is None:
            usage_usd = eff.get("spend_usd", 0.0)

        limit_usd = data.get("limit")
        if limit_usd is None:
            limit_usd = eff.get("limit_usd")

        remaining_usd = data.get("limit_remaining")
        if remaining_usd is None:
            remaining_usd = eff.get("remaining_usd")

        return {
            "provider": "openrouter",
            "usage_usd": float(usage_usd or 0.0),
            "limit_usd": float(limit_usd) if limit_usd is not None else None,
            "remaining_usd": float(remaining_usd) if remaining_usd is not None else None,
            "usage_daily_usd": float(data.get("usage_daily") or 0.0),
            "usage_monthly_usd": float(data.get("usage_monthly") or 0.0),
            "limit_reset": data.get("limit_reset"),
        }
    except Exception:
        return None


def estimate_image_input_tokens(width: int, height: int, prompt: str, num_calls: int = 1) -> int:
    # Estimate vision patch tokens (~260 base + tile scale) + text prompt tokens per call
    tiles_w = max(1, math.ceil(width / 512))
    tiles_h = max(1, math.ceil(height / 512))
    image_tokens = 256 + (tiles_w * tiles_h) * 85
    prompt_tokens = max(20, len(prompt) // 4)
    return image_tokens + prompt_tokens * num_calls


def enforce_demo_token_budget(image: Image.Image, prompt: str, is_spotting: bool = False) -> Image.Image:
    """
    Ensure a demo request stays within DEMO_MAX_INPUT_TOKENS (2,000 input tokens).
    If an uploaded image is very large, downscale it slightly in Demo Mode if needed,
    or reject if it still exceeds the 2,000 input token budget.
    """
    est = estimate_image_input_tokens(image.width, image.height, prompt, num_calls=2 if is_spotting else 1)
    if est > DEMO_MAX_INPUT_TOKENS:
        # Downscale to fit comfortably inside the 2,000 input token demo budget
        image = image.copy()
        image.thumbnail((1280, 1280))
        est = estimate_image_input_tokens(image.width, image.height, prompt, num_calls=2 if is_spotting else 1)
        if est > DEMO_MAX_INPUT_TOKENS:
            raise HttpError(
                400,
                f"Estimated input ({est} tokens) exceeds the Demo API limit of {DEMO_MAX_INPUT_TOKENS} input tokens per request. "
                "Please use a smaller image or set your own API key.",
            )
    return image


def get_client(base_url: str | None = None, api_key: str | None = None) -> OpenAI:
    resolved_key = (api_key if api_key is not None else get_api_key()).strip()
    resolved_url = (base_url if base_url is not None else get_base_url()).strip().rstrip("/")
    if not resolved_key:
        raise HttpError(500, "API key is not configured. Please set your own API key or configure .env.")
    return OpenAI(
        base_url=resolved_url,
        api_key=resolved_key,
        timeout=60.0,
        max_retries=2,
    )


def load_uploaded_image(file_storage) -> Image.Image:
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
    return image


def encode_image_base64(image: Image.Image, quality: int = 92) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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


def create_completion(client: OpenAI, messages: list[dict], stream: bool, model_name: str | None = None):
    target_model = (model_name or get_model_name()).strip()
    kwargs = {
        "model": target_model,
        "messages": messages,
        "temperature": 0.2 if stream else 0.0,
        "max_tokens": 8192,
        "top_p": 0.95,
        "stream": stream,
    }
    if stream:
        kwargs["stream_options"] = {"include_usage": True}

    for attempt in range(3):
        try:
            if stream and "stream_options" in kwargs:
                try:
                    return client.chat.completions.create(**kwargs)
                except APIError:
                    kwargs.pop("stream_options", None)
                    return client.chat.completions.create(**kwargs)
            return client.chat.completions.create(**kwargs)
        except (RateLimitError, APIConnectionError) as e:
            if attempt == 2:
                raise
            time.sleep(1.0 * (attempt + 1))


def extract_usage(usage_obj, elapsed_sec: float | None = None, fallback_text: str = "") -> dict:
    if usage_obj:
        total_time = getattr(usage_obj, "total_time", None)
        if total_time is None and elapsed_sec is not None:
            total_time = elapsed_sec
        prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
        total_tokens = getattr(usage_obj, "total_tokens", 0) or (prompt_tokens + completion_tokens)
        cost = getattr(usage_obj, "cost", None)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_time": round(float(total_time or 0.0), 2),
            "cost": float(cost) if cost is not None else None,
        }

    est_completion = max(1, len(fallback_text) // 4) if fallback_text else 0
    est_prompt = 320
    return {
        "prompt_tokens": est_prompt,
        "completion_tokens": est_completion,
        "total_tokens": est_prompt + est_completion,
        "total_time": round(float(elapsed_sec or 0.0), 2),
        "cost": None,
    }


def strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def parse_json(raw: str):
    text = strip_think(raw)
    text = re.sub(r'"label"\s*:\s*"text_content"\s*:', '"label":', text)
    text = re.sub(r'"text_content"\s*:\s*"label"\s*:', '"text_content":', text)

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
                    if isinstance(obj, dict) and ("box_2d" in obj or "bbox_2d" in obj or "bbox" in obj):
                        objects.append(obj)
                except Exception:
                    continue
            return objects if objects else None


def normalize_boxes(parsed, width: int, height: int, model_name: str = "") -> list[dict]:
    if isinstance(parsed, dict):
        parsed = next((v for v in parsed.values() if isinstance(v, list)), [])
    if not isinstance(parsed, list):
        return []

    default_yx = is_ymin_xmin_model(model_name)
    items = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        has_box_2d = "box_2d" in item
        coords = item.get("box_2d") or item.get("bbox_2d") or item.get("bbox")
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            continue
        try:
            c0, c1, c2, c3 = [float(c) for c in coords]
        except (TypeError, ValueError):
            continue

        text = str(item.get("text_content", item.get("text", item.get("label", ""))))

        if has_box_2d or default_yx:
            y1, x1, y2, x2 = c0, c1, c2, c3
        else:
            x1, y1, x2, y2 = c0, c1, c2, c3
            if len(text.strip()) >= 4 and (y2 - y1) > (x2 - x1) * 2.5 and (y2 > x2):
                x2, y2 = y2, x2

        items.append((text, [x1, y1, x2, y2]))

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
    elif max_value <= 1100.0:
        if "qwen" in model_name.lower():
            longest = float(max(width, height))
            q_sx = 1000.0 * width / longest
            q_sy = 1000.0 * height / longest
            max_x = max(max(c[0], c[2]) for _, c in items)
            max_y = max(max(c[1], c[3]) for _, c in items)
            if max_x <= q_sx * 1.05 and max_y <= q_sy * 1.05:
                sx, sy = q_sx, q_sy
            else:
                sx, sy = 1000.0, 1000.0
        else:
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


def compute_edge_map(image: Image.Image) -> tuple[list[float], Image.Image]:
    """Compute horizontal row edge energy and edge-filtered grayscale image."""
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    w, h = edges.size
    pixels = edges.load()
    row_energy = []
    for y in range(h):
        if y < 2 or y >= h - 2:
            row_energy.append(0.0)
            continue
        s = 0
        for x in range(2, w - 2, 2):
            v = pixels[x, y]
            if v > 28:
                s += v
        row_energy.append(s / max(1, (w // 2)))
    return row_energy, edges


def split_into_smart_strips(
    image: Image.Image, row_energy: list[float], max_strips_allowed: int = MAX_STRIPS
) -> list[tuple[int, int, Image.Image]]:
    """Slice tall images along wide paragraph whitespace valleys so multi-line blocks stay intact."""
    w, h = image.size
    if h <= 850 or max_strips_allowed <= 1:
        return [(0, h, image)]

    n_strips = min(max_strips_allowed, max(1, round(h / TARGET_STRIP_HEIGHT)))
    if n_strips <= 1:
        return [(0, h, image)]

    cuts = [0]
    step = h / n_strips
    search_radius = max(40, int(step * 0.35))

    for i in range(1, n_strips):
        target_y = int(round(i * step))
        y_min = max(cuts[-1] + 140, target_y - search_radius)
        y_max = min(h - 140, target_y + search_radius)
        best_y = target_y
        best_score = float("inf")
        for y in range(y_min, y_max + 1):
            window = row_energy[max(0, y - 7) : min(h, y + 8)]
            avg_energy = sum(window) / max(1, len(window))
            score = avg_energy + abs(y - target_y) * 0.002
            if score < best_score:
                best_score = score
                best_y = y
        cuts.append(best_y)
    cuts.append(h)

    strips = []
    for i in range(len(cuts) - 1):
        y0 = max(0, cuts[i] - (4 if i > 0 else 0))
        y1 = min(h, cuts[i + 1] + (4 if i < len(cuts) - 2 else 0))
        strips.append((y0, y1, image.crop((0, y0, w, y1))))
    return strips


def refine_boxes_sequence(edges: Image.Image, boxes: list[dict]) -> list[dict]:
    """
    Sequence-aware pixel refinement:
    1. Prevents Box 2 from ever jumping up onto Line 1's position when two lines are close.
    2. Prevents Box 1 from collapsing onto a thin top border/noise sliver above the text.
    """
    if not boxes:
        return []

    w, h = edges.size
    px = edges.load()

    y_limits = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b["box"]
        bh = max(0.006, y2 - y1)
        pad_y = min(0.025, bh * 0.28)
        low_y = max(0.0, y1 - pad_y)
        high_y = min(1.0, y2 + pad_y)

        for j, other in enumerate(boxes):
            if i == j:
                continue
            ox1, oy1, ox2, oy2 = other["box"]
            x_overlap = max(0.0, min(x2, ox2) - max(x1, ox1))
            if x_overlap <= 0.25 * min(max(1e-4, x2 - x1), max(1e-4, ox2 - ox1)):
                continue
            yc_i = (y1 + y2) / 2.0
            yc_j = (oy1 + oy2) / 2.0
            if yc_j < yc_i and oy2 > low_y:
                mid_boundary = (oy2 + y1) / 2.0 if oy2 <= y1 else (yc_j + yc_i) / 2.0
                low_y = max(low_y, mid_boundary)
            elif yc_j > yc_i and oy1 < high_y:
                mid_boundary = (y2 + oy1) / 2.0 if y2 <= oy1 else (yc_i + yc_j) / 2.0
                high_y = min(high_y, mid_boundary)

        y_limits.append((low_y, high_y))

    refined = []
    for i, b in enumerate(boxes):
        bx1, by1, bx2, by2 = b["box"]
        low_y, high_y = y_limits[i]

        x1 = int(round(bx1 * w))
        y1 = int(round(by1 * h))
        x2 = int(round(bx2 * w))
        y2 = int(round(by2 * h))
        bh = max(6, y2 - y1)

        sy1 = max(2, int(round(low_y * h)))
        sy2 = min(h - 2, int(round(high_y * h)))
        pad_x = max(4, min(16, int(bh * 0.35)))
        sx1 = max(2, x1 - pad_x)
        sx2 = min(w - 2, x2 + pad_x)

        if sx2 <= sx1 + 4 or sy2 <= sy1 + 4:
            refined.append(b)
            continue

        cw = sx2 - sx1
        ch = sy2 - sy1
        row_counts = []
        for r in range(ch):
            yy = sy1 + r
            c = 0
            for col in range(cw):
                if px[sx1 + col, yy] > 35:
                    c += 1
            row_counts.append(c)

        max_rc = max(row_counts) if row_counts else 0
        if max_rc < 4:
            refined.append(b)
            continue

        thresh_r = max(2, int(max_rc * 0.15))
        active_rows = [r for r, c in enumerate(row_counts) if c >= thresh_r]
        if not active_rows:
            refined.append(b)
            continue

        merge_gap = max(4, int(bh * 0.25))
        bands = []
        b_start = active_rows[0]
        b_prev = active_rows[0]
        for r in active_rows[1:]:
            if r - b_prev <= merge_gap:
                b_prev = r
            else:
                bands.append((b_start, b_prev))
                b_start = r
                b_prev = r
        bands.append((b_start, b_prev))

        min_valid_h = max(4, int(bh * 0.45))
        valid_bands = [bd for bd in bands if (bd[1] - bd[0] + 1) >= min_valid_h]
        candidate_bands = valid_bands if valid_bands else bands

        pred_center = (y1 + y2) / 2.0 - sy1
        best_band = min(candidate_bands, key=lambda bd: abs((bd[0] + bd[1]) / 2.0 - pred_center))
        ny1 = sy1 + best_band[0] - 2
        ny2 = sy1 + best_band[1] + 2

        if (ny2 - ny1) < int(bh * 0.6):
            ny1, ny2 = y1, y2

        refined.append(
            {
                "text": b["text"],
                "box": [
                    round(min(max(bx1, 0.0), 1.0), 5),
                    round(min(max(ny1 / h, 0.0), 1.0), 5),
                    round(min(max(bx2, 0.0), 1.0), 5),
                    round(min(max(ny2 / h, 0.0), 1.0), 5),
                ],
            }
        )

    return refined


def deduplicate_boxes(boxes: list[dict]) -> list[dict]:
    """Only deduplicate true duplicate detections from strip boundary overlaps."""
    merged = []
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        bh = max(1e-5, y2 - y1)
        bw = max(1e-5, x2 - x1)
        is_dup = False
        for m in merged:
            mx1, my1, mx2, my2 = m["box"]
            mbh = max(1e-5, my2 - my1)
            mbw = max(1e-5, mx2 - mx1)
            y_inter = max(0.0, min(y2, my2) - max(y1, my1))
            x_inter = max(0.0, min(x2, mx2) - max(x1, mx1))
            if (
                y_inter > 0.70 * min(bh, mbh)
                and x_inter > 0.50 * min(bw, mbw)
                and b["text"].strip().lower() == m["text"].strip().lower()
            ):
                is_dup = True
                break
        if not is_dup:
            merged.append(b)
    return merged


def stream_events(
    completion,
    start_time: float,
    demo_quota: dict | None,
    custom_auth: tuple[str, str] | None = None,
):
    if demo_quota:
        yield json.dumps({"type": "demo_quota", "data": demo_quota}) + "\n"

    buffer = ""
    emitted_text = ""
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
                        piece = buffer[:start]
                        emitted_text += piece
                        yield json.dumps({"type": "token", "text": piece}) + "\n"
                    buffer = buffer[start + len(open_tag) :]
                    in_think = True
                    continue
                safe = len(buffer)
                for i in range(1, len(open_tag)):
                    if buffer.endswith(open_tag[:i]):
                        safe = len(buffer) - i
                        break
                if safe > 0:
                    piece = buffer[:safe]
                    emitted_text += piece
                    yield json.dumps({"type": "token", "text": piece}) + "\n"
                buffer = buffer[safe:]
                break

    if buffer and not in_think:
        emitted_text += buffer
        yield json.dumps({"type": "token", "text": buffer}) + "\n"

    if not usage_data:
        usage_data = extract_usage(None, time.perf_counter() - start_time, fallback_text=emitted_text)

    yield json.dumps({"type": "usage", "data": usage_data}) + "\n"

    if custom_auth:
        key_quota = fetch_openrouter_key_quota(custom_auth[0], custom_auth[1])
        if key_quota:
            yield json.dumps({"type": "key_quota", "data": key_quota}) + "\n"


@app.get("/api/health")
def health():
    base_url, api_key, _, is_custom = get_request_config()
    key_quota = fetch_openrouter_key_quota(base_url, api_key) if (is_custom or not is_demo_mode_enabled()) else None
    return jsonify(
        {
            "status": "ok",
            "model": get_model_name(),
            "base_url": get_base_url(),
            "api_key_configured": bool(get_api_key()),
            "demo_mode_enabled": is_demo_mode_enabled(),
            "demo_quota": get_demo_quota_status(),
            "key_quota": key_quota,
        }
    )


@app.post("/api/ocr/full")
def ocr_full():
    base_url, api_key, model_name, is_custom = get_request_config()
    client = get_client(base_url=base_url, api_key=api_key)
    file_storage = request.files.get("file")
    image = load_uploaded_image(file_storage)

    demo_quota = None
    if not is_custom and is_demo_mode_enabled():
        image = enforce_demo_token_budget(image, FULL_PAGE_PROMPT, is_spotting=False)
        demo_quota = consume_demo_quota()

    base64_image = encode_image_base64(image)
    start_time = time.perf_counter()

    try:
        completion = create_completion(
            client,
            build_messages(base64_image, FULL_PAGE_PROMPT),
            stream=True,
            model_name=model_name,
        )
    except APIError as e:
        raise HttpError(502, f"API error: {e.message}")
    except Exception as e:
        raise HttpError(502, f"API error: {e}")

    custom_auth = (base_url, api_key) if (is_custom or not is_demo_mode_enabled()) else None

    def generator():
        try:
            for event in stream_events(completion, start_time, demo_quota, custom_auth=custom_auth):
                yield event
        except Exception as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return Response(stream_with_context(generator()), mimetype="application/x-ndjson")


@app.post("/api/ocr/spotting")
def ocr_spotting():
    base_url, api_key, model_name, is_custom = get_request_config()
    client = get_client(base_url=base_url, api_key=api_key)
    file_storage = request.files.get("file")
    image = load_uploaded_image(file_storage)
    prompt = get_spotting_prompt(model_name)

    demo_quota = None
    if not is_custom and is_demo_mode_enabled():
        image = enforce_demo_token_budget(image, prompt, is_spotting=True)
        demo_quota = consume_demo_quota()

    width, height = image.size
    row_energy, edges = compute_edge_map(image)
    # In Demo Mode, cap at 2 strips max so input tokens across the request stay <= 2,000
    max_strips = 2 if (not is_custom and is_demo_mode_enabled()) else MAX_STRIPS
    strips = split_into_smart_strips(image, row_energy, max_strips_allowed=max_strips)
    num_strips = len(strips)

    start_time = time.perf_counter()

    def process_strip(idx_and_strip):
        _, (y0, _, crop_img) = idx_and_strip
        b64 = encode_image_base64(crop_img)
        completion = create_completion(
            client,
            build_messages(b64, prompt),
            stream=False,
            model_name=model_name,
        )
        raw_text = completion.choices[0].message.content or ""
        strip_usage = extract_usage(getattr(completion, "usage", None), fallback_text=raw_text)
        cw, ch = crop_img.size
        local_boxes = normalize_boxes(parse_json(raw_text), cw, ch, model_name=model_name)
        global_boxes = []
        for b in local_boxes:
            lx1, ly1, lx2, ly2 = b["box"]
            gy1 = (y0 + ly1 * ch) / height
            gy2 = (y0 + ly2 * ch) / height
            global_boxes.append(
                {
                    "text": b["text"],
                    "box": [
                        round(min(max(lx1, 0.0), 1.0), 5),
                        round(min(max(gy1, 0.0), 1.0), 5),
                        round(min(max(lx2, 0.0), 1.0), 5),
                        round(min(max(gy2, 0.0), 1.0), 5),
                    ],
                }
            )
        return global_boxes, strip_usage

    try:
        with ThreadPoolExecutor(max_workers=min(4, num_strips)) as executor:
            strip_results = list(executor.map(process_strip, enumerate(strips)))
    except APIError as e:
        raise HttpError(502, f"API error: {e.message}")
    except Exception as e:
        raise HttpError(502, f"API error: {e}")

    elapsed = round(time.perf_counter() - start_time, 2)
    all_boxes = [b for group, _ in strip_results for b in group]
    total_prompt_tok = sum(u["prompt_tokens"] for _, u in strip_results)
    total_comp_tok = sum(u["completion_tokens"] for _, u in strip_results)
    costs = [u["cost"] for _, u in strip_results if u.get("cost") is not None]
    usage = {
        "prompt_tokens": total_prompt_tok,
        "completion_tokens": total_comp_tok,
        "total_tokens": total_prompt_tok + total_comp_tok,
        "total_time": elapsed,
        "cost": sum(costs) if costs else None,
    }

    deduped = deduplicate_boxes(all_boxes)
    boxes = refine_boxes_sequence(edges, deduped)
    key_quota = fetch_openrouter_key_quota(base_url, api_key) if (is_custom or not is_demo_mode_enabled()) else None

    return jsonify(
        {
            "boxes": boxes,
            "raw": json.dumps(boxes, indent=2, ensure_ascii=False),
            "width": width,
            "height": height,
            "usage": usage,
            "demo_quota": demo_quota or get_demo_quota_status(),
            "key_quota": key_quota,
        }
    )


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if FRONTEND_DIST and os.path.exists(FRONTEND_DIST):
        target = os.path.join(FRONTEND_DIST, path)
        if path and os.path.exists(target):
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({"status": "Backend running. Build frontend/dist to view UI."})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = "0.0.0.0" if os.getenv("PORT") else "127.0.0.1"
    app.run(host=host, port=port, debug=not bool(os.getenv("PORT")))
