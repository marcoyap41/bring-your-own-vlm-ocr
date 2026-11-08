# Bring Your Own VLM OCR (`bring-your-own-vlm-ocr`)

> **A provider-agnostic Vision OCR web app with real-time text streaming and interactive bounding-box text spotting, powered by any OpenAI-compatible Vision Language Model (VLM) endpoint.**

---

## Overview

Most modern OCR solutions lock you into proprietary vendor APIs or rigid local pipelines. **Bring Your Own VLM OCR** decouples the intelligence layer from the application layer:

- **Any Endpoint:** Works out-of-the-box with **OpenRouter**, **Groq**, **OpenAI**, **Together AI**, or local self-hosted runtimes like **Ollama**, **vLLM**, and **LM Studio**.
- **Dual OCR Modes:**
  - **Full Page OCR:** Live NDJSON token streaming for immediate document extraction.
  - **Interactive Text Spotting:** Line-level detection with normalized 2D bounding boxes overlaid dynamically on the image canvas.
- **Modern Minimalist Stack:** Built with Python 3.13 (`uv_build`), a lightweight Flask streaming backend, and a reactive Vite + React frontend.

---

## Architecture & Tech Stack

- **Backend:** Flask, Flask-CORS, Pillow (EXIF rotation & resize optimization), `openai` Python SDK (Chat Completions API with multimodal payloads).
- **Frontend:** React 19, Vite, custom SVG interactive bounding box overlay, clipboard integration, live token/request telemetry.
- **Package Manager:** `uv` by Astral for fast, reproducible Python packaging.

---

## Quickstart

### 1. Clone & Configure Environment

```powershell
# Copy template or create .env in project root
cp .env.example .env   # or edit .env directly
```

Configure your target provider in `.env`:

```env
# Cloud: OpenRouter (Default)
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=your_openrouter_api_key
MODEL_NAME=google/gemma-4-31b-it

# --- Alternative Examples ---
# Local Ollama:
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=ollama
# MODEL_NAME=minicpm-v

# Groq:
# OPENAI_BASE_URL=https://api.groq.com/openai/v1
# OPENAI_API_KEY=gsk_...
# MODEL_NAME=llama-3.2-11b-vision-preview
```

### 2. Install Dependencies

```powershell
# Backend (from project root)
uv sync

# Frontend
cd frontend
npm install
```

### 3. Run the Application

```powershell
# Terminal 1 - Backend (http://127.0.0.1:8000)
uv run python -m backend.main

# Terminal 2 - Frontend (http://localhost:5173)
cd frontend
npm run dev
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Returns backend status, active VLM model, base URL, and key configuration status. |
| `POST` | `/api/ocr/full` | Streams live extracted text tokens and telemetry as `application/x-ndjson`. |
| `POST` | `/api/ocr/spotting` | Returns detected text lines, raw model reasoning, and normalized `[x1, y1, x2, y2]` bounding coordinates. |

---

## License

MIT © [Marco Yap](https://github.com/marcoyap41)
