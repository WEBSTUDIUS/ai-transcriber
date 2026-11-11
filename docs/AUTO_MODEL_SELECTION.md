# Automatic local summary model selection

The local llama.cpp fallback selects a profile before each new model load, using `psutil.virtual_memory().available`, not just installed RAM. External OpenAI-compatible servers keep their own model selection.

| Profile | GGUF | Estimated inference RAM at 8K context and q8 KV |
| --- | --- | --- |
| 4b | Qwen3-4B-Instruct-2507 Q4_K_M | 3.5 GiB |
| 8b | Qwen3-8B Q4_K_M | 7 GiB |
| 14b | Qwen3-14B Q4_K_M | 11 GiB |

These are conservative planning estimates, not measured guarantees. Only the 4B model was evaluated for inference on the current 8 GB Mac; 8B/14B selection is covered by policy/integration tests. Model parameter count does not guarantee better summaries. 2B is excluded because it lost tasks in our Russian evaluation.

The budget is the smaller of:

- Total RAM minus the greater of 2 GiB or 35% of total RAM.
- Currently available RAM minus 0.5 GiB.

The largest fitting profile is selected. A relatively idle 8/16/32 GiB system therefore normally selects 4B/8B/14B, but memory pressure can reduce the choice. If even 4B does not fit, automatic startup stops before downloading or loading weights and explains how much memory is needed. Memory is checked again after download, before process launch. Retry after freeing memory to get a fresh selection. Missing memory statistics also stop automatic startup.

Larger contexts and f16 KV increase the estimated budget. Both client chunking and the server use the configured context. Dedicated CUDA VRAM is checked separately: insufficient VRAM disables GPU offloading and uses CPU/RAM. MPS shares system RAM. These checks reduce overload risk but are not an OS memory reservation and cannot prevent another application consuming memory afterward.

A model already loaded by this app is retained until it is unloaded; the UI reports its actual selected profile instead of suggesting a different one based on reduced free memory. New models are downloaded only when a summary actually needs the local fallback. Previously downloaded weights remain cached.

## Configuration

```dotenv
LLAMA_MODEL_PROFILE=auto
LLAMA_CTX=8192
LLAMA_KV_CACHE_TYPE=q8_0
```

`4b`, `8b` or `14b` pins a known profile while retaining the RAM check. `manual` requires both `LLAMA_MODEL_REPO` and `LLAMA_MODEL_FILE`; custom model memory use is unknown, so the automated estimate/check is bypassed. This is an explicit expert override, displayed as such. Legacy configurations with a repo/file pair and no profile retain manual behavior. An explicit `auto` ignores old pinned repo/file values.

## Интерфейс и поведение

В блоке саммари отображаются установленная и доступная RAM, бюджет, выбранная модель, контекст и причина выбора/отказа. Данные обновляются каждые 15 секунд, пока страница видима, при обновлении списка моделей и после завершения задания. Эндпоинт `/api/local-model` возвращает тот же план. Внешний сервер LLM может работать даже при нехватке памяти для локального запуска.

Автовыбор запускается после освобождения моделей распознавания, когда включена их выгрузка. При ручном выборе неизвестной модели оценка RAM не производится. 8B и 14B на этом Mac не загружались: проверена логика выбора, а реальное распознавание и саммари ранее проверены на 4B. Bitrix для этих проверок не использовался.
