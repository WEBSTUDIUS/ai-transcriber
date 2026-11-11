const $ = (id) => document.getElementById(id);

const fileInput = $("file");
const dropzone = $("dropzone");
const fileLabel = $("fileLabel");
const startButton = $("start");

const configKeys = [
    "backend", "device", "model", "language", "compute_type",
    "summary_prompt_type", "custom_prompt",
    "llm_endpoint", "llm_model", "llm_model_custom", "send_endpoint",
    "transcribe", "diarization", "summary", "send", "normalize",
    "unload_model"
];

const CUSTOM_SEND = "__custom__";
const LLM_CUSTOM = "__custom__";
const LLM_AUTO = "__auto__";

const TEXT_EXTENSIONS = new Set(["txt", "md", "log", "text"]);
const AUDIO_VIDEO_EXTENSIONS = new Set([
    "mp3", "wav", "m4a", "mp4", "mov", "avi", "mkv", "flac", "ogg", "opus", "aac", "wma", "webm"
]);

function setValue(id, value) {
    const el = $(id);
    if (!el) return;
    if (el.type === "checkbox") el.checked = !!value;
    else if (value !== undefined && value !== null) el.value = value;
}

function getValue(id) {
    const el = $(id);
    if (!el) return "";
    if (el.type === "checkbox") return el.checked;
    return el.value;
}

function getLLMModel() {
    const sel = $("llm_model");
    const custom = $("llm_model_custom");
    if (!sel) return "";
    if (sel.value === LLM_CUSTOM) return (custom ? custom.value.trim() : "");
    if (sel.value === LLM_AUTO) return "";
    return sel.value;
}

function setLLMModel(value) {
    const sel = $("llm_model");
    const custom = $("llm_model_custom");
    if (!sel) return;
    const v = (value || "").trim();
    const exists = [...sel.options].some((o) => o.value === v);
    if (v && exists) {
        sel.value = v;
        if (custom) custom.classList.add("hidden");
    } else if (v) {
        sel.value = LLM_CUSTOM;
        if (custom) {
            custom.classList.remove("hidden");
            custom.value = v;
        }
    } else {
        sel.value = LLM_AUTO;
        if (custom) {
            custom.classList.add("hidden");
            custom.value = "";
        }
    }
}

function applyConfig(config) {
    for (const key of configKeys) {
        if (key === "llm_model") continue; // селектор применим отдельно
        setValue(key, config[key]);
    }
    // LLM-селектор: сначала запомним сохранённое, список подтянется следом.
    window._pendingLLMModel = (config.llm_model || config.llm_model_custom || "").trim();
    setLLMModel(window._pendingLLMModel);
    updateDependencies();
}

async function saveConfig() {
    const data = {};
    for (const key of configKeys) {
        if (key === "llm_model") {
            data.llm_model = getLLMModel();
            continue;
        }
        data[key] = getValue(key);
    }
    try {
        await fetch("/api/config", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });
    } catch (_) {}
}

function getFileType(file) {
    if (!file) return null;
    const name = file.name || "";
    const ext = name.includes(".") ? name.split(".").pop().toLowerCase() : "";
    if (TEXT_EXTENSIONS.has(ext)) return "text";
    if (AUDIO_VIDEO_EXTENSIONS.has(ext)) return "media";
    return "unknown";
}

function setDisabled(id, disabled, title = "") {
    const el = $(id);
    if (!el) return;
    el.disabled = disabled;
    if (title) el.title = title;
}

function setAlert(id, message, type = "error") {
    const el = typeof id === "string" ? $(id) : id;
    if (!el) return;
    if (!message) {
        el.classList.add("hidden");
        el.textContent = "";
        return;
    }
    el.textContent = message;
    el.className = `alert ${type}`;
    el.classList.remove("hidden");
}

function showManualSend(show) {
    const wrap = $("send_endpoint_wrap");
    if (wrap) wrap.classList.toggle("hidden", !show);
}

function populateSendEndpoints() {
    const presets = window.AVAILABLE_SEND_ENDPOINTS || [];
    const sel = $("send_endpoint_preset");
    const manual = $("send_endpoint");
    const alertEl = $("send_endpoint_alert");
    if (!sel || !manual || !alertEl) return;

    sel.innerHTML = "";
    for (const preset of presets) {
        const opt = document.createElement("option");
        opt.value = preset.url;
        opt.textContent = preset.name;
        sel.appendChild(opt);
    }

    const customOpt = document.createElement("option");
    customOpt.value = CUSTOM_SEND;
    customOpt.textContent = "Custom / вручную";
    sel.appendChild(customOpt);

    const saved = manual.value.trim();

    if (presets.length === 0) {
        setAlert(
            alertEl,
            "Нет доступных вебхуков. Укажите URL вручную или задайте WEBHOOK_IT_TASK / WEBHOOK_HR / WEBHOOK_CALL_TRACK в .env."
        );
        sel.value = CUSTOM_SEND;
        showManualSend(true);
        return;
    }

    const match = presets.find((p) => p.url === saved);
    if (match) {
        sel.value = match.url;
        showManualSend(false);
    } else if (saved) {
        sel.value = CUSTOM_SEND;
        showManualSend(true);
    } else {
        sel.value = presets[0].url;
        manual.value = sel.value;
        showManualSend(false);
    }
    setAlert(alertEl, "");
}

function renderLocalModel(plan) {
    const box = $("localModelPlan");
    if (!box || !plan) return;
    const budgetLabel = plan.loaded ? "бюджет при старте" : "бюджет";
    const memory = plan.total_gib == null
        ? "RAM не определена"
        : `RAM ${plan.total_gib} ГиБ · доступно ${plan.available_gib} ГиБ · ${budgetLabel} ${plan.budget_gib} ГиБ`;
    const mode = plan.mode === "auto" ? "Автовыбор" : "Ручной выбор";
    box.className = `alert ${plan.fits ? "info" : "error"}`;
    box.textContent = `${mode} локального саммари: ${plan.label}. ${memory}. Контекст ${plan.context}. ${plan.reason}`;
}

async function refreshLocalModel() {
    try {
        const response = await fetch("/api/local-model");
        if (response.ok) renderLocalModel(await response.json());
    } catch (_) { /* Keep the initial server-rendered plan during a disconnect. */ }
}

renderLocalModel(window.INITIAL_LOCAL_MODEL);
setInterval(() => {
    if (document.visibilityState === "visible") void refreshLocalModel();
}, 15000);

async function refreshLLMModels() {
    void refreshLocalModel();
    const sel = $("llm_model");
    const custom = $("llm_model_custom");
    const alertEl = $("llm_model_alert");
    if (!sel || !alertEl) return;

    const pending = (window._pendingLLMModel || "").trim();
    const current = pending || getLLMModel();
    window._pendingLLMModel = "";

    if (!$("summary").checked) {
        sel.innerHTML = "";
        const opt = document.createElement("option");
        opt.value = LLM_AUTO;
        opt.textContent = "Саммари выключено";
        sel.appendChild(opt);
        sel.value = LLM_AUTO;
        if (custom) custom.classList.add("hidden");
        setAlert(alertEl, "");
        return;
    }

    const endpoint = $("llm_endpoint").value.trim();
    if (!endpoint) {
        sel.innerHTML = "";
        const opt = document.createElement("option");
        opt.value = LLM_AUTO;
        opt.textContent = "Укажите endpoint";
        sel.appendChild(opt);
        sel.value = LLM_AUTO;
        setAlert(alertEl, "Укажите LLM endpoint.", "error");
        return;
    }

    setAlert(alertEl, "Загрузка моделей...", "info");
    sel.disabled = true;

    try {
        const response = await fetch(
            `/api/llm/models?base_url=${encodeURIComponent(endpoint)}`
        );
        const data = await response.json();
        const models = data.models || [];

        sel.innerHTML = "";

        const autoOpt = document.createElement("option");
        autoOpt.value = LLM_AUTO;
        autoOpt.textContent = "Авто — сервер или локальная модель";
        sel.appendChild(autoOpt);

        for (const id of models) {
            const opt = document.createElement("option");
            opt.value = id;
            opt.textContent = id;
            sel.appendChild(opt);
        }

        const customOpt = document.createElement("option");
        customOpt.value = LLM_CUSTOM;
        customOpt.textContent = "Ввести вручную...";
        sel.appendChild(customOpt);

        if (data.error) {
            setLLMModel(current);
            if (data.fallback_model) {
                setAlert(alertEl, data.fallback_ready === false
                    ? `Сервер недоступен. Локальный запуск приостановлен: ${data.fallback_reason}`
                    : `Сервер недоступен. При обработке автоматически запустится ${data.fallback_model}.`,
                    data.fallback_ready === false ? "error" : "info");
            } else {
                setAlert(alertEl, `Модели недоступны (${data.error}). Можно ввести ID вручную.`, "error");
            }
            return;
        }
        if (!models.length) {
            setLLMModel(current);
            setAlert(alertEl, "Нет доступных моделей. Проверьте endpoint или введите ID вручную.", "error");
            return;
        }

        // Восстанавливаем сохранённый выбор, иначе — Auto.
        if (current && [...sel.options].some((o) => o.value === current)) {
            sel.value = current;
            if (custom) custom.classList.add("hidden");
        } else if (current) {
            sel.value = LLM_CUSTOM;
            if (custom) {
                custom.classList.remove("hidden");
                custom.value = current;
            }
            setAlert(alertEl, `Сохранённая модель "${current}" не найдена на эндпоинте. Доступно: ${models.length}.`, "error");
            return;
        } else {
            sel.value = LLM_AUTO;
            if (custom) custom.classList.add("hidden");
        }
        window._pendingLLMModel = "";
        setAlert(alertEl, `Доступно моделей: ${models.length}`, "info");
    } catch (error) {
        sel.innerHTML = "";
        const autoOpt = document.createElement("option");
        autoOpt.value = LLM_AUTO;
        autoOpt.textContent = "Авто — сервер или локальная модель";
        sel.appendChild(autoOpt);
        const customOpt = document.createElement("option");
        customOpt.value = LLM_CUSTOM;
        customOpt.textContent = "Ввести вручную...";
        sel.appendChild(customOpt);
        setLLMModel(current);
        setAlert(alertEl, `Ошибка получения моделей: ${error.message}`, "error");
    } finally {
        sel.disabled = !$("summary").checked;
        updateDependencies();
    }
}

function updateModelHint() {
    const hint = $("model_hint");
    const model = $("model");
    if (!hint || !model) return;
    const texts = {
        tiny: "39M · ~1 ГБ RAM · смоки / слабый сервер",
        base: "74M · ~1-2 ГБ RAM · черновики",
        small: "244M · ~2-3 ГБ RAM · сервер 4 ГБ",
        medium: "769M · ~4-6 ГБ RAM · нужно 8+ ГБ",
        "large-v3": "1550M · ~10-16 ГБ RAM · макс. точность",
        "large-v3-turbo": "809M · ~5-8 ГБ RAM · дефолт, в 8× быстрее large-v3",
    };
    hint.textContent = texts[model.value] || "";
}

function updateFileTypeUI(file) {
    const type = getFileType(file);
    const isText = type === "text";
    const isMedia = type === "media";

    // Transcription is mandatory for media and meaningless for text transcripts.
    setDisabled("transcribe", isText || isMedia, isText ? "This file is already a text transcript." : "Media files require transcription.");
    $("transcribe").checked = isMedia;

    setDisabled("model", !isMedia, isText ? "Not applicable to text files." : "");
    setDisabled("backend", !isMedia, isText ? "Not applicable to text files." : "");
    setDisabled("device", !isMedia, isText ? "Not applicable to text files." : "");
    setDisabled("language", !isMedia, isText ? "Not applicable to text files." : "");
    setDisabled("compute_type", !isMedia, isText ? "Not applicable to text files." : "");
    setDisabled("diarization", !isMedia, isText ? "Diarization requires audio/video transcription." : "");
    setDisabled("normalize", !isMedia, isText ? "Audio normalization requires an audio/video file." : "");

    if (isText) {
        $("diarization").checked = false;
        $("normalize").checked = false;
        if (!$("summary").checked && !$("send").checked) $("summary").checked = true;
    }

    if (isMedia) {
        $("transcribe").checked = true;
    }

    updateDependencies();
}

function updateDependencies() {
    const type = getFileType(fileInput.files[0]);
    const isText = type === "text";
    const isMedia = type === "media";
    const transcribe = $("transcribe").checked;
    if ($("send").checked) $("summary").checked = true;

    // These options are only meaningful as part of media transcription.
    setDisabled("diarization", !isMedia || !transcribe, "Diarization is part of transcription and requires audio/video.");
    setDisabled("normalize", !isMedia || !transcribe, "Normalization requires audio/video transcription.");

    if (!isMedia || !transcribe) {
        $("diarization").checked = false;
        $("normalize").checked = false;
    }

    const summary = $("summary").checked;
    setDisabled("summary_prompt_type", !summary);
    setDisabled("custom_prompt", !summary || $("summary_prompt_type").value !== "custom");
    setDisabled("llm_endpoint", !summary);
    setDisabled("llm_model", !summary);
    setDisabled("llm_model_custom", !summary || $("llm_model").value !== LLM_CUSTOM);
    setDisabled("llm_refresh", !summary);
    setDisabled("unload_model", !summary);

    const send = $("send").checked;
    setDisabled("send_endpoint_preset", !send);
    setDisabled("send_endpoint", !send);
    if (!send) setAlert("send_endpoint_alert", "");

    // A media file without transcription has no transcript to summarize/send.
    if (isMedia && !transcribe) {
        setDisabled("summary", true, "Transcription is required before summary.");
        setDisabled("send", true, "Transcription is required before sending.");
    } else {
        setDisabled("summary", false);
        setDisabled("send", false);
    }
}

function updateFileLabel(file) {
    fileLabel.replaceChildren();
    const name = document.createElement("span");
    name.textContent = file ? file.name : "Перетащите файл сюда";
    const detail = document.createElement("small");
    detail.textContent = file
        ? `${(file.size / 1024 / 1024).toFixed(2)} МБ · ${getFileType(file)}`
        : "Аудио, видео или текст · до 2 ГБ · нажмите для выбора";
    fileLabel.append(name, document.createElement("br"), detail);
}

$("summary_prompt_type").addEventListener("change", updateDependencies);

applyConfig(window.INITIAL_CONFIG || {});
updateModelHint();
updateDependencies();
populateSendEndpoints();
refreshLLMModels();

for (const id of ["transcribe", "summary", "send", "diarization", "normalize"]) {
    $(id).addEventListener("change", updateDependencies);
}

$("summary").addEventListener("change", refreshLLMModels);

let llmModelsTimer = null;
$("llm_endpoint").addEventListener("input", () => {
    clearTimeout(llmModelsTimer);
    llmModelsTimer = setTimeout(refreshLLMModels, 600);
});
$("llm_refresh").addEventListener("click", refreshLLMModels);
$("llm_model").addEventListener("change", () => {
    const custom = $("llm_model_custom");
    if ($("llm_model").value === LLM_CUSTOM) {
        if (custom) custom.classList.remove("hidden");
    } else if (custom) {
        custom.classList.add("hidden");
        custom.value = "";
    }
    window._pendingLLMModel = "";
    updateDependencies();
});
$("model").addEventListener("change", updateModelHint);

$("send_endpoint_preset").addEventListener("change", () => {
    const sel = $("send_endpoint_preset");
    const manual = $("send_endpoint");
    if (sel.value === CUSTOM_SEND) {
        showManualSend(true);
        manual.value = "";
    } else {
        showManualSend(false);
        manual.value = sel.value;
    }
});

const DEVICE_COMPUTE = { cuda: "float16", mps: "auto", cpu: "auto", auto: "auto" };

$("device").addEventListener("change", () => {
    $("compute_type").value = DEVICE_COMPUTE[$("device").value] || "auto";
});

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    updateFileLabel(file);
    updateFileTypeUI(file);
});

["dragenter", "dragover"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
    });
});

dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length) {
        fileInput.files = files;
        fileInput.dispatchEvent(new Event("change"));
    }
});

function renderJob(job) {
    $("progressCard").classList.remove("hidden");
    const progress = job.status === "completed" ? 100 : (job.progress || 0);
    $("progressBar").style.width = `${progress}%`;
    $("progressText").textContent = `${progress}%`;
    $("currentStep").textContent = job.status === "completed" ? "Готово" : (job.current_step || job.status);
    $("logs").textContent = (job.logs || []).join("\n");

    if (job.status === "running" || job.status === "queued") {
        $("statusBadge").textContent = "В ОБРАБОТКЕ";
    } else if (job.status === "completed") {
        $("statusBadge").textContent = "ГОТОВО";
    } else if (job.status === "failed") {
        $("statusBadge").textContent = "ОШИБКА";
    }

    if (job.status === "failed" && job.error) {
        $("currentStep").textContent = `Ошибка: ${job.error}`;
    }
}

function renderResultLinks(jobId, job) {
    const links = [];
    if (job.transcript_file) {
        links.push(`<a class="downloadLink" href="/api/jobs/${jobId}/file?kind=transcript" download>⬇ Скачать транскрипцию</a>`);
    }
    if (job.summary_file) {
        links.push(`<a class="downloadLink" href="/api/jobs/${jobId}/file?kind=summary" download>⬇ Скачать саммари</a>`);
    }
    $("resultLinks").innerHTML = links.join("");
}

async function pollJob(jobId) {
    try {
        const response = await fetch(`/api/jobs/${jobId}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const job = await response.json();
        renderJob(job);

        if (job.status === "queued" || job.status === "running") {
            setTimeout(() => pollJob(jobId), 700);
            return;
        }

        startButton.disabled = false;
        void refreshLocalModel();
        renderResultLinks(jobId, job);
    } catch (error) {
        startButton.disabled = false;
        $("statusBadge").textContent = "ОШИБКА";
        $("currentStep").textContent = `Ошибка получения статуса: ${error.message}`;
    }
}

startButton.addEventListener("click", async () => {
    const file = fileInput.files[0];
    if (!file) {
        alert("Сначала выберите аудио, видео или текстовый файл.");
        return;
    }

    const type = getFileType(file);
    if (type === "unknown") {
        alert("Формат не поддерживается. Выберите аудио, видео или TXT/MD/LOG.");
        return;
    }

    if (file.size === 0 || file.size > 2 * 1024 ** 3) {
        alert("Выберите непустой файл размером до 2 ГБ.");
        return;
    }
    if ($("send").checked && (!$("summary").checked || !getValue("send_endpoint").trim())) {
        alert("Для отправки включите саммари и укажите вебхук.");
        return;
    }
    updateFileTypeUI(file);

    const form = new FormData();
    form.append("file", file);
    for (const key of configKeys) {
        if (key === "llm_model") {
            form.append("llm_model", getLLMModel());
            continue;
        }
        if (key === "llm_model_custom") {
            form.append("llm_model_custom", String(getValue("llm_model_custom")));
            continue;
        }
        form.append(key, String(getValue(key)));
    }

    startButton.disabled = true;
    $("statusBadge").textContent = "ЗАПУСК";
    $("resultLinks").innerHTML = "";
    $("logs").textContent = "";
    $("progressCard").classList.remove("hidden");
    $("progressBar").style.width = "0%";
    $("progressText").textContent = "0%";
    $("currentStep").textContent = "Запуск...";

    await saveConfig();

    try {
        const response = await fetch("/api/jobs", { method: "POST", body: form });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Failed to create job");
        pollJob(data.job_id);
    } catch (error) {
        startButton.disabled = false;
        $("statusBadge").textContent = "ОШИБКА";
        $("currentStep").textContent = `Ошибка: ${error.message}`;
        alert(error.message);
    }
});
