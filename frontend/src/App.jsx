import { useCallback, useEffect, useRef, useState } from "react";

const MODES = [
  { id: "full", label: "Full Page OCR", hint: "Extract all text from the document, streamed live." },
  { id: "spotting", label: "Text Spotting", hint: "Detect each text line and its position on the image." },
];

const ACCEPTED = ["image/jpeg", "image/png", "image/webp"];
const STORAGE_KEY = "byo_vlm_ocr_custom_api";
const MODE_STORAGE_KEY = "byo_vlm_ocr_api_mode";
const CLIENT_ID_KEY = "byo_vlm_ocr_client_id";
const GITHUB_REPO_URL = "https://github.com/marcoyap41/bring-your-own-vlm-ocr";

const PROVIDER_PRESETS = [
  { label: "OpenRouter", baseUrl: "https://openrouter.ai/api/v1", sampleModel: "google/gemma-4-31b-it" },
  { label: "Groq", baseUrl: "https://api.groq.com/openai/v1", sampleModel: "meta-llama/llama-4-scout-17b-16e-instruct" },
  { label: "OpenAI", baseUrl: "https://api.openai.com/v1", sampleModel: "gpt-4o-mini" },
  { label: "Together AI", baseUrl: "https://api.together.xyz/v1", sampleModel: "Qwen/Qwen2.5-VL-72B-Instruct" },
];

function getOrCreateClientId() {
  try {
    let cid = localStorage.getItem(CLIENT_ID_KEY);
    if (!cid) {
      cid =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID()
          : `cid-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
      localStorage.setItem(CLIENT_ID_KEY, cid);
    }
    return cid;
  } catch {
    return "ephemeral-client";
  }
}

function loadSavedApiConfig() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (parsed && parsed.apiKey && parsed.baseUrl && parsed.model) {
      return parsed;
    }
  } catch {
    // ignore storage errors
  }
  return null;
}

function loadSavedApiMode(hasSavedPersonal) {
  try {
    const savedMode = localStorage.getItem(MODE_STORAGE_KEY);
    if (savedMode === "personal" && hasSavedPersonal) return "personal";
    if (savedMode === "demo") return "demo";
  } catch {
    // ignore storage errors
  }
  return hasSavedPersonal ? "personal" : "demo";
}

function formatNumber(val) {
  if (val === null || val === undefined) return "-";
  return Number(val).toLocaleString();
}

function formatUsd(val) {
  if (val === null || val === undefined) return "-";
  const num = Number(val);
  if (num === 0) return "0.00";
  if (num < 0.01) return num.toFixed(4);
  if (num < 1) return num.toFixed(4);
  return num.toFixed(2);
}

function renderQuotaTooltipContent(baseUrl) {
  const cleanUrl = (baseUrl || "").trim().replace(/\/+$/, "");
  const lower = cleanUrl.toLowerCase();

  if (lower.includes("openrouter.ai")) {
    return (
      <>
        Fetched live from OpenRouter&apos;s key endpoint (<code>GET /api/v1/key</code>). Displays all-time credit usage
        and the credit limit set on this specific API key.
      </>
    );
  }

  if (
    lower.includes("localhost") ||
    lower.includes("127.0.0.1") ||
    lower.includes("0.0.0.0") ||
    lower.includes("192.168.") ||
    lower.includes("://10.")
  ) {
    return (
      <>
        Running locally or self-hosted. There are no cloud credit quotas or billing limits for this endpoint.
      </>
    );
  }

  if (
    lower.includes("api.openai.com") ||
    lower.includes("api.groq.com") ||
    lower.includes("api.together.xyz") ||
    lower.includes("generativelanguage.googleapis.com")
  ) {
    return (
      <>
        Credit balance endpoint is not publicly available for this provider via standard API keys. Please check billing
        and credit limits on your provider&apos;s web console.
      </>
    );
  }

  return (
    <>
      Attempted via standard key introspection endpoint (<code>GET {cleanUrl || "/api/v1"}/key</code>). If unsupported
      by your provider, check billing directly on your provider dashboard.
    </>
  );
}

async function readError(response) {
  try {
    const data = await response.json();
    return data.detail || `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [mode, setMode] = useState("full");
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [fullText, setFullText] = useState("");
  const [boxes, setBoxes] = useState([]);
  const [rawJson, setRawJson] = useState("");
  const [showRaw, setShowRaw] = useState(false);
  const [activeIndex, setActiveIndex] = useState(null);
  const [copied, setCopied] = useState(null);
  const [health, setHealth] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [baseSize, setBaseSize] = useState(null);
  const [panning, setPanning] = useState(false);

  // Usage & Demo Quota State
  const [clientId] = useState(() => getOrCreateClientId());
  const [usage, setUsage] = useState(null);
  const [sessionTokens, setSessionTokens] = useState(0);
  const [sessionCost, setSessionCost] = useState(0);
  const [sessionRequests, setSessionRequests] = useState(0);
  const [demoQuota, setDemoQuota] = useState({
    limit_requests: 5,
    used_requests: 0,
    remaining_requests: 5,
    max_input_tokens: 2000,
  });
  const [keyQuota, setKeyQuota] = useState(null);

  // BYOK / API Mode State
  const [savedPersonalApi, setSavedPersonalApi] = useState(() => loadSavedApiConfig());
  const [apiMode, setApiMode] = useState(() => loadSavedApiMode(Boolean(loadSavedApiConfig())));
  const [showApiModal, setShowApiModal] = useState(false);
  const [modalTab, setModalTab] = useState("personal");
  const [formBaseUrl, setFormBaseUrl] = useState("");
  const [formApiKey, setFormApiKey] = useState("");
  const [formModel, setFormModel] = useState("");
  const [showKeyText, setShowKeyText] = useState(false);
  const [formError, setFormError] = useState("");

  const inputRef = useRef(null);
  const abortRef = useRef(null);
  const stageRef = useRef(null);
  const imgRef = useRef(null);
  const lineRefs = useRef([]);
  const panStartRef = useRef({ x: 0, y: 0, scrollLeft: 0, scrollTop: 0, moved: false });

  const isServerDemoMode = health?.demo_mode_enabled ?? true;
  const isDemoMode = isServerDemoMode && (apiMode === "demo" || !savedPersonalApi);
  const activeCustomApi = (!isServerDemoMode && !savedPersonalApi)
    ? null
    : (!isDemoMode ? savedPersonalApi : null);
  const demoBaseUrl = health?.base_url || "https://openrouter.ai/api/v1";
  const demoModel = health?.model || "google/gemma-4-31b-it";
  const activeBaseUrl = activeCustomApi?.baseUrl || demoBaseUrl;
  const activeModel = activeCustomApi?.model || demoModel;
  const isApiReady = !isDemoMode ? (activeCustomApi?.apiKey ? true : Boolean(health?.api_key_configured)) : Boolean(health?.api_key_configured);

  const applyUsage = (usageData) => {
    if (!usageData) return;
    setUsage(usageData);
    setSessionTokens((prev) => prev + (usageData.total_tokens || 0));
    if (typeof usageData.cost === "number" && !isNaN(usageData.cost)) {
      setSessionCost((prev) => prev + usageData.cost);
    }
    setSessionRequests((prev) => prev + 1);
  };

  const openApiModal = () => {
    setFormBaseUrl(savedPersonalApi?.baseUrl || "");
    setFormApiKey(savedPersonalApi?.apiKey || "");
    setFormModel(savedPersonalApi?.model || "");
    setFormError("");
    setShowKeyText(false);
    setModalTab("personal");
    setShowApiModal(true);
  };

  const handleSaveCustomApi = (e) => {
    e.preventDefault();
    const trimmedUrl = formBaseUrl.trim().replace(/\/+$/, "");
    const trimmedKey = formApiKey.trim();
    const trimmedModel = formModel.trim();

    if (!trimmedUrl) {
      setFormError("Please enter an API Base URL.");
      return;
    }
    if (!trimmedKey) {
      setFormError("Please enter your API Key.");
      return;
    }
    if (!trimmedModel) {
      setFormError("Please enter a Model ID.");
      return;
    }

    const nextConfig = {
      baseUrl: trimmedUrl,
      apiKey: trimmedKey,
      model: trimmedModel,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(nextConfig));
    localStorage.setItem(MODE_STORAGE_KEY, "personal");
    setSavedPersonalApi(nextConfig);
    setApiMode("personal");
    setShowApiModal(false);
  };

  const handleUseSavedPersonalApi = () => {
    if (!savedPersonalApi) return;
    localStorage.setItem(MODE_STORAGE_KEY, "personal");
    setApiMode("personal");
    setShowApiModal(false);
  };

  const handleClearPersonalApi = () => {
    localStorage.removeItem(STORAGE_KEY);
    setSavedPersonalApi(null);
    setFormBaseUrl("");
    setFormApiKey("");
    setFormModel("");
    // Fall back to demo mode
    localStorage.setItem(MODE_STORAGE_KEY, "demo");
    setApiMode("demo");
    setKeyQuota(null);
  };

  const handleApplyDemoApi = () => {
    if (demoQuota.remaining_requests <= 0) return;
    localStorage.setItem(MODE_STORAGE_KEY, "demo");
    setApiMode("demo");
    setKeyQuota(null);
    setShowApiModal(false);
  };

  const clampZoom = useCallback((val) => Math.min(4, Math.max(0.5, Math.round(val * 100) / 100)), []);

  const updateBaseSize = useCallback(() => {
    const img = imgRef.current;
    const stage = stageRef.current;
    if (!img || !stage || !img.naturalWidth || !img.naturalHeight) return;
    const availW = Math.max(100, stage.clientWidth - 24);
    const availH = Math.max(100, stage.clientHeight - 24);
    const ratio = Math.min(1, availW / img.naturalWidth, availH / img.naturalHeight);
    setBaseSize({
      w: Math.round(img.naturalWidth * ratio),
      h: Math.round(img.naturalHeight * ratio),
    });
  }, []);

  useEffect(() => {
    window.addEventListener("resize", updateBaseSize);
    return () => window.removeEventListener("resize", updateBaseSize);
  }, [updateBaseSize]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !previewUrl) return;
    const onWheel = (e) => {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.15 : -0.15;
      setZoom((prev) => clampZoom(prev + delta));
    };
    stage.addEventListener("wheel", onWheel, { passive: false });
    return () => stage.removeEventListener("wheel", onWheel);
  }, [previewUrl, clampZoom]);

  useEffect(() => {
    const headers = { "X-Client-Id": clientId };
    if (activeCustomApi) {
      headers["X-Api-Key"] = activeCustomApi.apiKey;
      headers["X-Base-Url"] = activeCustomApi.baseUrl;
      headers["X-Model-Name"] = activeCustomApi.model;
    }
    fetch("/api/health", { headers })
      .then((r) => r.json())
      .then((data) => {
        setHealth(data);
        if (data?.demo_quota) {
          setDemoQuota(data.demo_quota);
        }
        setKeyQuota(data?.key_quota || null);
      })
      .catch(() => setHealth({ status: "offline" }));
  }, [clientId, activeCustomApi]);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const resetResults = () => {
    setFullText("");
    setBoxes([]);
    setRawJson("");
    setError("");
    setActiveIndex(null);
  };

  const selectFile = useCallback((selected) => {
    if (!selected) return;
    if (!ACCEPTED.includes(selected.type)) {
      setError("Please upload a JPG, PNG, or WEBP image.");
      return;
    }
    abortRef.current?.abort();
    setLoading(false);
    resetResults();
    setZoom(1);
    setBaseSize(null);
    setFile(selected);
  }, []);

  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
      if (item) selectFile(item.getAsFile());
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [selectFile]);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    selectFile(e.dataTransfer.files?.[0]);
  };

  const run = async () => {
    if (!file || loading) return;
    resetResults();
    setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;
    const form = new FormData();
    form.append("file", file);

    const headers = {
      "X-Client-Id": clientId,
    };
    if (activeCustomApi) {
      headers["X-Api-Key"] = activeCustomApi.apiKey;
      headers["X-Base-Url"] = activeCustomApi.baseUrl;
      headers["X-Model-Name"] = activeCustomApi.model;
    }

    try {
      const response = await fetch(`/api/ocr/${mode}`, {
        method: "POST",
        headers,
        body: form,
        signal: controller.signal,
      });
      if (!response.ok) {
        if (response.status === 429 && isDemoMode) {
          setDemoQuota((prev) => ({
            ...prev,
            used_requests: prev.limit_requests,
            remaining_requests: 0,
          }));
        }
        throw new Error(await readError(response));
      }

      if (mode === "full") {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        const handleLine = (line) => {
          const trimmed = line.trim();
          if (!trimmed) return;
          try {
            const event = JSON.parse(trimmed);
            if (event.type === "token") {
              setFullText((prev) => prev + (event.text || ""));
            } else if (event.type === "usage") {
              applyUsage(event.data);
            } else if (event.type === "demo_quota") {
              setDemoQuota(event.data);
            } else if (event.type === "key_quota") {
              setKeyQuota(event.data);
            } else if (event.type === "error") {
              setError(event.detail || "Stream error");
            }
          } catch {
            setFullText((prev) => prev + line);
          }
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";
          for (const line of lines) {
            handleLine(line);
          }
        }
        if (buffer.trim()) {
          handleLine(buffer);
        }
      } else {
        const data = await response.json();
        setBoxes(data.boxes || []);
        setRawJson(data.raw || "");
        if (data.usage) applyUsage(data.usage);
        if (data.demo_quota) setDemoQuota(data.demo_quota);
        if (data.key_quota) setKeyQuota(data.key_quota);
        if (!data.boxes?.length) setError("The model did not return any readable bounding boxes.");
      }
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
      abortRef.current = null;
    }
  };

  const stop = () => abortRef.current?.abort();

  const copy = async (text, key) => {
    await navigator.clipboard.writeText(text);
    setCopied(key);
    setTimeout(() => setCopied(null), 1200);
  };

  const clearImage = () => {
    abortRef.current?.abort();
    setFile(null);
    setPreviewUrl(null);
    setZoom(1);
    setBaseSize(null);
    resetResults();
  };

  const onStageMouseDown = (e) => {
    if (e.button !== 0 || !stageRef.current) return;
    setPanning(true);
    panStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      scrollLeft: stageRef.current.scrollLeft,
      scrollTop: stageRef.current.scrollTop,
      moved: false,
    };
  };

  const onStageMouseMove = (e) => {
    if (!panning || !stageRef.current) return;
    const dx = e.clientX - panStartRef.current.x;
    const dy = e.clientY - panStartRef.current.y;
    if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
      panStartRef.current.moved = true;
    }
    stageRef.current.scrollLeft = panStartRef.current.scrollLeft - dx;
    stageRef.current.scrollTop = panStartRef.current.scrollTop - dy;
  };

  const onStageMouseUp = () => {
    setPanning(false);
  };

  const hasResult = fullText || boxes.length > 0;
  const currentMode = MODES.find((m) => m.id === mode);

  const demoOnlyRatio = Math.max(
    0,
    Math.min(100, ((demoQuota?.remaining_requests ?? 5) / (demoQuota?.limit_requests || 5)) * 100)
  );

  const activeQuotaRatio = isDemoMode
    ? demoOnlyRatio
    : keyQuota
      ? keyQuota.limit_usd
        ? Math.max(0, Math.min(100, (keyQuota.usage_usd / keyQuota.limit_usd) * 100))
        : 100
      : 0;

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="logo" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 7V5a2 2 0 0 1 2-2h2" />
              <path d="M17 3h2a2 2 0 0 1 2 2v2" />
              <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
              <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
              <line x1="7" y1="9" x2="17" y2="9" />
              <line x1="7" y1="13" x2="14" y2="13" />
              <line x1="7" y1="17" x2="11" y2="17" />
            </svg>
          </div>
          <div>
            <div className="brand-title-row">
              <h1>Bring-Your-Own-VLM OCR</h1>
              <span className="brand-tag">BYOK Vision Engine</span>
            </div>
            <p>Extract and locate text in images with any OpenAI-compatible Vision-Language Model</p>
          </div>
        </div>
      </header>

      <section className="api-bar">
        <div className="api-bar-info">
          <div className="api-bar-row">
            <span className="api-bar-label">Current API:</span>
            <code className="api-bar-code">{activeBaseUrl}</code>
            <span className={`api-mode-badge ${isDemoMode ? "demo" : "personal"}`}>
              {isDemoMode ? "Free Demo API" : "Personal API"}
            </span>
          </div>
          <div className="api-bar-row">
            <span className="api-bar-label">Model:</span>
            <code className="api-bar-code">{activeModel}</code>
          </div>
        </div>
        <button className="primary api-bar-btn" onClick={openApiModal}>
          Set your own API
        </button>
      </section>

      <section className="metrics-bar">
        <div className="metric-card">
          <div className="metric-row">
            <span className="metric-label">Last Request Usage</span>
            {!isDemoMode && usage?.cost !== null && usage?.cost !== undefined && (
              <span className="metric-cost-tag">${formatUsd(usage.cost)}</span>
            )}
          </div>
          <div className="metric-value">
            {usage ? `${formatNumber(usage.total_tokens)} tokens` : "-"}
          </div>
          <span className="metric-sub">
            {usage
              ? `In: ${formatNumber(usage.prompt_tokens)} / Out: ${formatNumber(usage.completion_tokens)} (${usage.total_time}s)`
              : isDemoMode
                ? `Demo cap: max ${formatNumber(demoQuota.max_input_tokens)} input tok/req`
                : "No requests yet"}
          </span>
        </div>

        <div className="metric-card">
          <div className="metric-row">
            <span className="metric-label">Session Total</span>
            {!isDemoMode && sessionCost > 0 && (
              <span className="metric-cost-tag">${formatUsd(sessionCost)}</span>
            )}
          </div>
          <div className="metric-value">{formatNumber(sessionTokens)} tokens</div>
          <span className="metric-sub">
            Cumulative in current tab ({sessionRequests} {sessionRequests === 1 ? "req" : "reqs"})
          </span>
        </div>

        <div className="metric-card">
          <div className="metric-row">
            <span className="metric-label metric-label-with-info">
              {isDemoMode ? "Demo Request Quota" : "Personal API Credit Usage"}
              {!isDemoMode && (
                <span className="info-tooltip-wrap" tabIndex={0} aria-label="How credit usage is fetched">
                  <span className="info-icon">?</span>
                  <span className="info-tooltip">{renderQuotaTooltipContent(activeBaseUrl)}</span>
                </span>
              )}
            </span>
            <span className="metric-pct">
              {!isDemoMode
                ? !keyQuota
                  ? "N/A"
                  : keyQuota.limit_usd === null
                    ? "No Cap"
                    : `${activeQuotaRatio.toFixed(1).replace(/\.0$/, "")}%`
                : `${activeQuotaRatio.toFixed(1).replace(/\.0$/, "")}%`}
            </span>
          </div>
          <div className="metric-value">
            {isDemoMode
              ? `${demoQuota.remaining_requests} / ${demoQuota.limit_requests} remaining`
              : keyQuota
                ? keyQuota.limit_usd !== null
                  ? `$${formatUsd(keyQuota.usage_usd)} / $${formatUsd(keyQuota.limit_usd)} USD`
                  : `$${formatUsd(keyQuota.usage_usd)} / ∞ USD`
                : "Not Available"}
          </div>
          <div className="progress-track">
            <div
              className={`progress-fill ${
                (isDemoMode && demoQuota.remaining_requests === 0) ||
                (!isDemoMode && keyQuota?.remaining_usd !== null && keyQuota?.remaining_usd <= 0)
                  ? "exhausted"
                  : ""
              }`}
              style={{ width: `${activeQuotaRatio}%` }}
            />
          </div>
          <span className="metric-sub">
            {isDemoMode
              ? demoQuota.remaining_requests > 0
                ? `${demoQuota.used_requests} used · Max ${formatNumber(demoQuota.max_input_tokens)} input tok/req`
                : "Demo limit reached — set your own API"
              : keyQuota
                ? keyQuota.remaining_usd !== null
                  ? `Remaining: $${formatUsd(keyQuota.remaining_usd)} USD`
                  : `All-time usage: $${formatUsd(keyQuota.usage_usd)} USD`
                : "Check billing directly on your provider dashboard"}
          </span>
        </div>
      </section>

      {showApiModal && (
        <div className="modal-backdrop" onClick={() => setShowApiModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-active-indicator">
                <span className="modal-active-label">Active Config:</span>
                <span className={`api-mode-badge ${isDemoMode ? "demo" : "personal"}`}>
                  {isDemoMode ? "Free Demo API" : "Personal API"}
                </span>
              </div>
              <button className="ghost small" onClick={() => setShowApiModal(false)}>
                Close
              </button>
            </div>

            <div className="tabs modal-tabs">
              <button
                type="button"
                className={`tab ${modalTab === "personal" ? "active" : ""}`}
                onClick={() => setModalTab("personal")}
              >
                Personal API
              </button>
              <button
                type="button"
                className={`tab ${modalTab === "demo" ? "active" : ""}`}
                onClick={() => setModalTab("demo")}
              >
                Demo API
              </button>
            </div>

            {modalTab === "personal" ? (
              <div className="modal-section">
                <div className="config-summary-box">
                  <div className="config-summary-header">
                    <span className="config-summary-title">Saved Personal API</span>
                    <button
                      type="button"
                      className="ghost small icon-btn danger-text"
                      onClick={handleClearPersonalApi}
                      disabled={!savedPersonalApi}
                      title="Clear saved Personal API"
                      aria-label="Clear saved Personal API"
                    >
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                        <path d="M10 11v6" />
                        <path d="M14 11v6" />
                        <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                      </svg>
                    </button>
                  </div>
                  <div className="config-summary-row">
                    <span className="api-bar-label">API:</span>
                    <code className="api-bar-code">{savedPersonalApi?.baseUrl || "-"}</code>
                  </div>
                  <div className="config-summary-row">
                    <span className="api-bar-label">Model:</span>
                    <code className="api-bar-code">{savedPersonalApi?.model || "-"}</code>
                  </div>
                  <button
                    type="button"
                    className="primary small config-box-btn"
                    onClick={handleUseSavedPersonalApi}
                    disabled={!savedPersonalApi || !isDemoMode}
                  >
                    {savedPersonalApi && !isDemoMode ? "In Use" : "Use This Config"}
                  </button>
                </div>

                <form className="modal-form" onSubmit={handleSaveCustomApi}>
                  <div className="preset-row">
                    <span className="preset-label">Quick Presets:</span>
                    {PROVIDER_PRESETS.map((p) => (
                      <button
                        key={p.label}
                        type="button"
                        className="ghost small"
                        onClick={() => {
                          setFormBaseUrl(p.baseUrl);
                          setFormModel(p.sampleModel);
                        }}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>

                  <label className="form-field">
                    <span>API Base URL</span>
                    <input
                      type="url"
                      placeholder="https://openrouter.ai/api/v1"
                      value={formBaseUrl}
                      onChange={(e) => setFormBaseUrl(e.target.value)}
                      required
                    />
                  </label>

                  <label className="form-field">
                    <span>API Key</span>
                    <div className="key-input-wrap">
                      <input
                        type={showKeyText ? "text" : "password"}
                        placeholder="sk-or-v1-..."
                        value={formApiKey}
                        onChange={(e) => setFormApiKey(e.target.value)}
                        required
                      />
                      <button
                        type="button"
                        className="ghost small"
                        onClick={() => setShowKeyText((v) => !v)}
                      >
                        {showKeyText ? "Hide" : "Show"}
                      </button>
                    </div>
                  </label>

                  <label className="form-field">
                    <span>Model</span>
                    <input
                      type="text"
                      placeholder="google/gemma-4-31b-it"
                      value={formModel}
                      onChange={(e) => setFormModel(e.target.value)}
                      required
                    />
                    <span className="field-hint">
                      Must be a Vision-capable model (VLM) that supports image inputs (e.g. Gemma 4, GPT-4o, Qwen2.5-VL).
                    </span>
                  </label>

                  {formError && <div className="error">{formError}</div>}

                  <button type="submit" className="primary modal-full-btn">
                    Save &amp; Use Personal API
                  </button>
                </form>

                <div className="byok-info">
                  <strong>BYOK &amp; Security</strong>
                  <p>
                    Your API key is saved only in your browser&apos;s local storage and forwarded over HTTPS strictly
                    in-memory per request. We never store or log your key on our server. For 100% local isolation, set
                    this app on your own machine{" "}
                    <a href={GITHUB_REPO_URL} target="_blank" rel="noreferrer">
                      here
                    </a>
                    .
                  </p>
                </div>
              </div>
            ) : (
              <div className="modal-section">
                <div className="config-summary-box">
                  <span className="config-summary-title">Free-Limited Demo API</span>
                  <div className="config-summary-row">
                    <span className="api-bar-label">API:</span>
                    <code className="api-bar-code">{demoBaseUrl}</code>
                  </div>
                  <div className="config-summary-row">
                    <span className="api-bar-label">Model:</span>
                    <code className="api-bar-code">{demoModel}</code>
                  </div>
                </div>

                <div className="byok-info">
                  <strong>About Free Demo API</strong>
                  <p>
                    This built-in demo API lets you try the app immediately without setting up an account. Usage is
                    limited to <strong>{demoQuota.limit_requests} requests</strong> per user and{" "}
                    <strong>{formatNumber(demoQuota.max_input_tokens)} input tokens</strong> per request.
                  </p>
                </div>

                <div className="metric-card modal-quota-card">
                  <div className="metric-row">
                    <span className="metric-label">Remaining Request Usage / Limit</span>
                    <span className="metric-pct">{demoOnlyRatio.toFixed(0)}%</span>
                  </div>
                  <div className="metric-value">
                    {demoQuota.remaining_requests} / {demoQuota.limit_requests} remaining
                  </div>
                  <div className="progress-track">
                    <div
                      className={`progress-fill ${demoQuota.remaining_requests === 0 ? "exhausted" : ""}`}
                      style={{ width: `${demoOnlyRatio}%` }}
                    />
                  </div>
                  <span className="metric-sub">
                    {demoQuota.remaining_requests > 0
                      ? `${demoQuota.used_requests} used · Max ${formatNumber(demoQuota.max_input_tokens)} input tok/req`
                      : "Demo quota exhausted — please switch to Personal API"}
                  </span>
                </div>

                <button
                  type="button"
                  className="primary modal-full-btn"
                  onClick={handleApplyDemoApi}
                  disabled={demoQuota.remaining_requests <= 0 || isDemoMode}
                >
                  {demoQuota.remaining_requests <= 0
                    ? "Demo Quota Exhausted"
                    : isDemoMode
                      ? "In Use"
                      : "Use Demo API"}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      <main className="layout">
        <section className="panel image-panel">
          {!previewUrl ? (
            <div
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
            >
              <div className="drop-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 19V5M5 12l7-7 7 7" />
                </svg>
              </div>
              <h2>Drop an image here</h2>
              <p>
                or click to browse / paste with <kbd>Ctrl</kbd> + <kbd>V</kbd>
              </p>
              <div className="format-pills">
                <span className="format-pill">JPG</span>
                <span className="format-pill">PNG</span>
                <span className="format-pill">WEBP</span>
              </div>
            </div>
          ) : (
            <>
              <div className="image-toolbar">
                <span className="filename" title={file?.name}>{file?.name || "Pasted image"}</span>
                <div className="toolbar-actions">
                  <div className="zoom-controls">
                    <button
                      className="ghost small"
                      onClick={() => setZoom((z) => clampZoom(z - 0.25))}
                      disabled={zoom <= 0.5}
                      title="Zoom out"
                    >
                      −
                    </button>
                    <button
                      className="ghost small zoom-readout"
                      onClick={() => setZoom(1)}
                      title="Reset zoom to 100%"
                    >
                      {Math.round(zoom * 100)}%
                    </button>
                    <button
                      className="ghost small"
                      onClick={() => setZoom((z) => clampZoom(z + 0.25))}
                      disabled={zoom >= 4}
                      title="Zoom in"
                    >
                      +
                    </button>
                  </div>
                  <button className="ghost" onClick={() => inputRef.current?.click()}>Replace</button>
                  <button className="ghost" onClick={clearImage}>Remove</button>
                </div>
              </div>
              <div
                ref={stageRef}
                className={`image-stage ${zoom > 1 ? "zoomed" : ""} ${panning ? "panning" : ""}`}
                onMouseDown={onStageMouseDown}
                onMouseMove={onStageMouseMove}
                onMouseUp={onStageMouseUp}
                onMouseLeave={onStageMouseUp}
                onDoubleClick={() => setZoom((z) => (z === 1 ? 2 : 1))}
              >
                <div className="image-wrap">
                  <img
                    ref={imgRef}
                    src={previewUrl}
                    alt="Uploaded"
                    draggable={false}
                    onLoad={updateBaseSize}
                    style={
                      baseSize
                        ? {
                            width: `${Math.round(baseSize.w * zoom)}px`,
                            height: `${Math.round(baseSize.h * zoom)}px`,
                            maxWidth: "none",
                            maxHeight: "none",
                          }
                        : undefined
                    }
                  />
                  {boxes.length > 0 && (
                    <svg className="overlay" viewBox="0 0 1 1" preserveAspectRatio="none">
                      {boxes.map((b, i) => (
                        <rect
                          key={i}
                          x={b.box[0]}
                          y={b.box[1]}
                          width={Math.max(b.box[2] - b.box[0], 0.001)}
                          height={Math.max(b.box[3] - b.box[1], 0.001)}
                          className={activeIndex === i ? "active" : ""}
                          vectorEffect="non-scaling-stroke"
                          onMouseEnter={() => setActiveIndex(i)}
                          onMouseLeave={() => setActiveIndex(null)}
                          onClick={(e) => {
                            if (panStartRef.current.moved) return;
                            e.stopPropagation();
                            setActiveIndex(i);
                            lineRefs.current[i]?.scrollIntoView({
                              behavior: "smooth",
                              block: "nearest",
                            });
                            copy(b.text, `box-${i}`);
                          }}
                        />
                      ))}
                    </svg>
                  )}
                  {loading && <div className="scanline" />}
                </div>
              </div>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED.join(",")}
            hidden
            onChange={(e) => {
              selectFile(e.target.files?.[0]);
              e.target.value = "";
            }}
          />
        </section>

        <section className="panel result-panel">
          <div className="tabs">
            {MODES.map((m) => (
              <button
                key={m.id}
                className={`tab ${mode === m.id ? "active" : ""}`}
                onClick={() => {
                  if (loading) return;
                  setMode(m.id);
                  resetResults();
                }}
                disabled={loading}
              >
                {m.label}
              </button>
            ))}
          </div>
          <p className="hint">{currentMode.hint}</p>

          <div className="actions">
            {loading ? (
              <button className="primary danger" onClick={stop}>Stop</button>
            ) : (
              <button
                className="primary"
                onClick={run}
                disabled={!file || (isDemoMode && demoQuota.remaining_requests <= 0)}
              >
                {mode === "full" ? "Extract Text" : "Spot Text"}
              </button>
            )}
            {loading && (
              <span className="spinner-label">
                <span className="spinner" /> {mode === "full" ? "Reading..." : "Locating text..."}
              </span>
            )}
          </div>

          {error && <div className="error">{error}</div>}

          <div className="result">
            {!hasResult && !loading && !error && (
              <div className="empty">
                {file ? "Ready. Choose a mode and run it." : "Upload an image to get started."}
              </div>
            )}

            {mode === "full" && (fullText || loading) && (
              <div className="text-result">
                <div className="result-header">
                  <span>Extracted text</span>
                  {fullText && !loading && (
                    <button className="ghost small" onClick={() => copy(fullText, "full")}>
                      {copied === "full" ? "Copied" : "Copy"}
                    </button>
                  )}
                </div>
                <pre className={loading ? "streaming" : ""}>{fullText}</pre>
              </div>
            )}

            {mode === "spotting" && boxes.length > 0 && (
              <div className="lines-result">
                <div className="result-header">
                  <span>{boxes.length} lines detected</span>
                  <div className="toolbar-actions">
                    <button className="ghost small" onClick={() => setShowRaw((v) => !v)}>
                      {showRaw ? "Hide JSON" : "Show JSON"}
                    </button>
                    <button className="ghost small" onClick={() => copy(boxes.map((b) => b.text).join("\n"), "all")}>
                      {copied === "all" ? "Copied" : "Copy all"}
                    </button>
                  </div>
                </div>
                {showRaw ? (
                  <pre className="raw">{rawJson}</pre>
                ) : (
                  <ul className="lines">
                    {boxes.map((b, i) => (
                      <li
                        key={i}
                        ref={(el) => {
                          lineRefs.current[i] = el;
                        }}
                        className={activeIndex === i ? "active" : ""}
                        onMouseEnter={() => setActiveIndex(i)}
                        onMouseLeave={() => setActiveIndex(null)}
                        onClick={() => copy(b.text, `box-${i}`)}
                      >
                        <span className="index">{i + 1}</span>
                        <span className="line-text">{b.text || <em>(empty)</em>}</span>
                        <span className="copy-tag">{copied === `box-${i}` ? "Copied" : "Copy"}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
