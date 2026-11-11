$ErrorActionPreference = "Stop"
uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Override the Windows CUDA source with CPU wheels in this environment.
uv pip install `
    torch==2.8.0 `
    torchvision==0.23.0 `
    torchaudio==2.8.0 `
    --index-url https://download.pytorch.org/whl/cpu
exit $LASTEXITCODE
