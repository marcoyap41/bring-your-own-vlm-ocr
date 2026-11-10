# Bring-Your-Own-VLM OCR (`bring-your-own-vlm-ocr`)

> **A provider-agnostic Vision OCR web application with real-time text streaming and interactive bounding-box text spotting, powered by any OpenAI-compatible Vision-Language Model (VLM).**

---

## Overview

Most modern OCR solutions lock you into proprietary vendor APIs or rigid local pipelines. **Bring-Your-Own-VLM OCR** decouples the intelligence layer from the application layer:

- **Any VLM Endpoint:** Works seamlessly with **OpenRouter**, **Groq**, **OpenAI**, **Together AI**, or self-hosted runtimes like **Ollama**, **vLLM**, and **LM Studio**.
- **Dual OCR Extraction Modes:**
  - **Full Page OCR:** Live NDJSON token streaming for immediate markdown/text extraction.
  - **Interactive Text Spotting:** Line-level detection with normalized 2D bounding boxes overlaid dynamically on the image canvas.
- **Computer Vision Post-Processing:** Automated edge-energy detection and horizontal contrast refinement to align bounding boxes tightly to actual text lines.
- **Smart Image Tiling & Parallel Processing:** Splits tall documents along blank line margins and processes slices in parallel (`ThreadPoolExecutor`), eliminating VLM hallucination on high-resolution text.

---

## Key Features

### 1. In-App BYOK & Multi-Provider Switcher
- Switch between providers on the fly directly in the web UI without restarting the backend.
- 1-click presets for **OpenRouter**, **Groq**, **OpenAI**, **Ollama**, and **LM Studio**.
- Keys configured in the web UI are stored securely in your browser's `localStorage` and sent directly to your local backend — never shared with third-party servers.

### 2. Dual Deployment Modes (Personal vs. Public Demo)
- **Personal Local Mode (Default):** When you clone the repository and configure `.env`, the backend provides full, unlimited OCR requests using your configured provider.
- **Public Demo Mode (`ENABLE_DEMO_MODE=true`):** Enforces a per-client quota (5 free requests) and automatic image downscaling to prevent quota abuse when hosting public demos.

### 3. Smart Strip Splitting & Multi-Format Coordinate Normalization
- Automatically handles diverse VLM bounding-box formats:
  - Gemma / PaliGemma: `[ymin, xmin, ymax, xmax]` normalized to 0–1000.
  - Generic / Qwen / OpenAI: `[xmin, ymin, xmax, ymax]`.
- Post-OCR Canny edge scanning snaps bounding boxes precisely to text contours.

### 4. Real-Time Token & Provider Usage Monitoring
- Live token breakdown per request: prompt tokens, completion tokens, execution time, and estimated cost.
- Cumulative session totals tracking.
- Live credit introspection for supported providers (e.g. OpenRouter balance and daily limits).

---

## Quickstart

### 1. Clone & Configure Environment

```powershell
# Copy template into .env in the project root
cp .env.example .env
```

> **Vision-Capable Model (VLM) Required:** The selected model **must** be a multimodal Vision-Language Model supporting image inputs (e.g., `google/gemma-4-31b-it`, `qwen2.5-vl:7b`, `gpt-4o`, `llama-3.2-11b-vision-preview`). Standard text-only models (e.g. `gpt-3.5-turbo`, text-only `llama-3`) will fail on image payload processing.

Edit `.env` to set your default Demo API provider:

```env
# Cloud: OpenRouter (Default)
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=your_openrouter_api_key
MODEL_NAME=google/gemma-4-31b-it

# Set to true only if hosting a rate-limited public showcase:
ENABLE_DEMO_MODE=false
```

#### Alternative Provider Examples:

```env
# Local Ollama (100% Offline & Private):
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
MODEL_NAME=qwen2.5-vl:7b

# Groq Vision:
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...
MODEL_NAME=llama-3.2-11b-vision-preview

# OpenAI:
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-proj-...
MODEL_NAME=gpt-4o
```

### 2. Install Dependencies

```powershell
# Backend (from project root using uv)
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

| Method | Endpoint | Description | Custom Headers |
|---|---|---|---|
| `GET` | `/api/health` | Returns backend status, active VLM model, and quota mode. | `X-Api-Key`, `X-Base-Url` (optional) |
| `POST` | `/api/ocr/full` | Streams live extracted text tokens as `application/x-ndjson`. | `X-Api-Key`, `X-Base-Url`, `X-Model-Name` |
| `POST` | `/api/ocr/spotting` | Returns detected text lines, refined bounding coordinates, and token metrics. | `X-Api-Key`, `X-Base-Url`, `X-Model-Name` |
