$ErrorActionPreference = "Stop"
uv sync
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# Override the Windows CUDA source with CPU wheels in this environment.
uv pip install `
    torch==2.8.0+cpu `
    torchvision==0.23.0+cpu `
    torchaudio==2.8.0+cpu `
    --no-sources `
    --default-index https://download.pytorch.org/whl/cpu
exit $LASTEXITCODE
