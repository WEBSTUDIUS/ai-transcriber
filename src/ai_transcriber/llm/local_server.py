
from __future__ import annotations

import atexit
import os
import platform
import signal
import subprocess
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable

import requests
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TotalFileSizeColumn,
    TransferSpeedColumn,
)

from ai_transcriber.config import (
    LLAMA_CACHE_DIR,
    LLAMA_GPU_LAYERS,
    LLAMA_PORT,
    LLAMA_KV_CACHE_TYPE,
)

from .model_selection import configured_selection, detect_memory, memory_budget, GIB

LLAMA_CPP_REPO = "ggml-org/llama.cpp"
DEFAULT_RELEASE = "b11985"


#
# How long to wait (seconds) for a llama-server that another process is
# already starting (e.g. first-run model download) before giving up and
# starting our own.
#
START_WAIT_TIMEOUT = 1800.0


def _platform_spec() -> tuple[str, str]:
    """
    Returns (platform_key, archive_suffix) for the current machine.

    llama.cpp release assets are named like:
        llama-b<build>-bin-<platform><arch>.<ext>
    """
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
        return f"macos-{arch}", ".tar.gz"

    if system == "linux":
        arch = "x64" if machine in ("x86_64", "amd64") else "arm64"
        return f"ubuntu-{arch}", ".tar.gz"

    if system == "windows":
        arch = "x64" if machine in ("x86_64", "amd64") else "arm64"
        return f"win-cpu-{arch}", ".zip"

    raise RuntimeError(
        f"llama.cpp fallback is not supported on this platform: "
        f"{system}/{machine}"
    )


def _resolve_asset(platform_key: str, suffix: str) -> tuple[str, str]:
    """
    Resolve the latest llama.cpp release tag and the download URL of
    the matching binary archive. Falls back to a pinned release if the
    GitHub API is unreachable.
    """
    try:
        response = requests.get(
            f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases?per_page=10",
            timeout=10,
        )
        response.raise_for_status()
        for release in response.json():
            tag = release["tag_name"]
            match = next(
                (asset["browser_download_url"] for asset in release.get("assets", [])
                 if asset["name"].startswith(f"llama-{tag}-bin-{platform_key}")
                 and asset["name"].endswith(suffix)),
                None,
            )
            if match:
                return tag, match
    except (requests.RequestException, KeyError, ValueError):
        pass

    tag = DEFAULT_RELEASE
    return tag, (
        f"https://github.com/{LLAMA_CPP_REPO}/releases/download/{tag}/"
        f"llama-{tag}-bin-{platform_key}{suffix}"
    )


def _gpu_layers_for_memory(requested: int, required_gib: float) -> int:
    """Dedicated GPU VRAM is separate from the host RAM used by the policy."""
    import torch
    if requested and torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            if free < (required_gib + .5) * GIB:
                return 0
        except (RuntimeError, AttributeError):
            return 0
    return requested


class LocalLLMServer:
    """
    Downloads llama.cpp and a GGUF model (configured GGUF model) and
    runs llama-server as an OpenAI-compatible endpoint on localhost.

    Used by the CLI as a fallback when LM Studio is not reachable.
    """

    def __init__(
        self,
        console: Console | None = None,
        log: Callable[[str], None] | None = None,
        port: int = LLAMA_PORT,
        ctx_size: int | None = None,
        ngpu: int = LLAMA_GPU_LAYERS,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        self.console = console or Console()
        self.log = log
        self.on_progress = on_progress
        self.port = port
        self.selection = configured_selection(context=ctx_size)
        self.ctx_size = self.selection.profile.context
        self.ngpu = ngpu
        self.cache_dir = LLAMA_CACHE_DIR
        self.binary_path: Path | None = None
        self.model_path: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._adopted = False

    def _check_memory(self) -> None:
        if self.selection.mode == 'manual':
            return
        available = detect_memory()
        if available is None or memory_budget(available) < self.selection.profile.required_gib * GIB:
            raise RuntimeError(
                f"Недостаточно доступной RAM для {self.selection.profile.label}. "
                f"Нужен бюджет {self.selection.profile.required_gib:.1f} ГиБ. "
                "Закройте лишние программы и повторите запуск."
            )

    def _log(self, message: str) -> None:
        if self.log is not None:
            self.log(message)
        else:
            self.console.print(message)

    def _report_progress(self, percent: float) -> None:
        if self.on_progress is not None:
            self.on_progress(max(0.0, min(100.0, percent)))

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def model_name(self) -> str:
        return self.selection.profile.filename

    @staticmethod
    def _binary_name() -> str:
        return "llama-server.exe" if os.name == "nt" else "llama-server"

    def ensure_binary(self) -> None:
        # Prefer an already-downloaded llama.cpp binary: upstream publishes
        # new releases often, but re-downloading each time is wasteful.
        existing = sorted(
            (self.cache_dir / "llama.cpp").rglob(self._binary_name())
        )
        if existing:
            self.binary_path = existing[-1]
            return

        platform_key, suffix = _platform_spec()
        release, url = _resolve_asset(platform_key, suffix)
        dest_dir = self.cache_dir / "llama.cpp" / release
        bin_path = dest_dir / self._binary_name()

        dest_dir.mkdir(parents=True, exist_ok=True)
        archive = dest_dir / f"llama-{release}-bin-{platform_key}{suffix}"

        if not archive.exists():
            self._log(f"Downloading llama.cpp ({release})...")
            self._download(
                url,
                archive,
                f"llama.cpp {release}",
                progress_range=(0, 15),
            )

        self._log("Extracting llama.cpp...")
        self._extract(archive, dest_dir)

        if not bin_path.exists():
            hits = sorted(dest_dir.rglob(self._binary_name()))
            if not hits:
                raise RuntimeError(
                    "llama-server binary not found in downloaded archive."
                )
            bin_path = hits[0]

        if os.name != "nt":
            bin_path.chmod(bin_path.stat().st_mode | 0o111)

        self.binary_path = bin_path

    def ensure_model(self) -> None:
        model_path = self.cache_dir / "models" / self.model_name

        # The supported multi-billion-parameter GGUFs must exceed 100 MB.
        if model_path.exists() and model_path.stat().st_size > 100 * 1024 * 1024:
            self.model_path = model_path
            return

        if model_path.exists():
            try:
                model_path.unlink()
            except OSError:
                pass

        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/{self.selection.profile.repo}/"
            f"resolve/main/{self.model_name}"
        )

        self._log(
            f"Downloading model {self.selection.profile.repo}:{self.model_name} "
            "(one-time download)..."
        )
        self._download(
            url,
            model_path,
            self.model_name,
            progress_range=(15, 95),
        )
        self.model_path = model_path

    def prepare(self) -> None:
        self._check_memory()
        self._log(f"Local model: {self.selection.profile.label}; {self.selection.reason}")
        self.ensure_binary()
        self.ensure_model()
        self._report_progress(96)

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return

        if not self.binary_path or not self.model_path:
            raise RuntimeError(
                "Call prepare() before start()."
            )

        self._check_memory()
        if self.selection.mode != 'manual':
            layers = _gpu_layers_for_memory(self.ngpu, self.selection.profile.required_gib)
            if layers != self.ngpu:
                self._log("Dedicated GPU memory is insufficient; running the local LLM on CPU.")
                self.ngpu = layers
        cmd = [
            str(self.binary_path),
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.ctx_size),
            "--reasoning",
            "off",
            "--parallel",
            "1",
            "--cache-type-k",
            LLAMA_KV_CACHE_TYPE,
            "--cache-type-v",
            LLAMA_KV_CACHE_TYPE,
            "--flash-attn",
            "on",
        ]
        # Pass zero explicitly: omitting this flag lets llama.cpp offload by default.
        cmd += ["--n-gpu-layers", str(self.ngpu)]

        self._log(
            f"Starting llama-server on port {self.port} "
            f"(ctx_size={self.ctx_size}, ngpu={self.ngpu})..."
        )

        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
        atexit.register(self.stop)
        self._write_pid_file()

    def wait_ready(self, timeout: float = 300) -> None:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{self.port}/health"

        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    "llama-server exited before becoming ready."
                )
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    self._report_progress(100)
                    self._log("llama-server is ready.")
                    return
            except requests.RequestException:
                pass
            time.sleep(1)

        raise RuntimeError(
            "llama-server did not become ready in time."
        )

    def prepare_and_start(self) -> None:
        if self._reuse_or_wait():
            return
        self.prepare()
        self.start()
        self.wait_ready()

    @property
    def _pidfile(self) -> Path:
        return self.cache_dir / "llama-server.pid"

    def _server_healthy(self) -> bool:
        try:
            response = requests.get(
                f"http://127.0.0.1:{self.port}/health",
                timeout=2,
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _read_pid_file(self) -> tuple[int | None, int | None]:
        if not self._pidfile.exists():
            return None, None
        try:
            lines = self._pidfile.read_text(encoding="utf-8").split()
            owner_pid = int(lines[0]) if lines else None
            server_pid = int(lines[1]) if len(lines) > 1 else None
            return owner_pid, server_pid
        except (OSError, ValueError):
            return None, None

    def _write_pid_file(self) -> None:
        if self._proc is None:
            return
        try:
            self._pidfile.parent.mkdir(parents=True, exist_ok=True)
            self._pidfile.write_text(
                f"{os.getpid()}\n{self._proc.pid}\n",
                encoding="utf-8",
            )
        except OSError:
            pass

    def _remove_pid_file(self) -> None:
        try:
            self._pidfile.unlink(missing_ok=True)
        except OSError:
            pass

    def _reuse_or_wait(self) -> bool:
        """
        Return True if a llama-server is already (or becomes) available,
        so we reuse it instead of spawning a duplicate process.

        Coordination across processes uses a pidfile plus the /health
        endpoint:
          * live owner + healthy server      -> reuse it
          * live owner + server still starting -> wait until it is ready
          * dead owner + orphaned server     -> reclaim it (free port + RAM)
          * nothing listening on the port    -> start our own server
        """
        owner_pid, server_pid = self._read_pid_file()

        if owner_pid is not None and self._is_pid_alive(owner_pid):
            if self._server_healthy():
                self._log("Reusing already-running llama-server.")
                self._adopted = True
                return True

            self._log(
                "Another llama-server is starting; "
                "waiting for it to become ready..."
            )
            deadline = time.monotonic() + START_WAIT_TIMEOUT
            while time.monotonic() < deadline:
                if self._server_healthy():
                    self._log(
                        "Reusing the llama-server started earlier."
                    )
                    self._adopted = True
                    return True
                if not self._is_pid_alive(owner_pid):
                    self._log(
                        "The starting process is gone; "
                        "starting our own server."
                    )
                    self._pidfile.unlink(missing_ok=True)
                    return False
                time.sleep(2)

            self._log(
                "Timed out waiting for the other llama-server; "
                "starting our own."
            )
            return False

        if owner_pid is not None:
            # Stale pidfile: the original owner is gone. Reclaim the
            # orphaned server so the port and RAM are released.
            if (
                server_pid is not None
                and self._is_pid_alive(server_pid)
                and self._server_healthy()
            ):
                self._reclaim(server_pid)
            else:
                self._pidfile.unlink(missing_ok=True)

        if self._server_healthy():
            self._log(
                "Reusing llama-server already listening on the port."
            )
            self._adopted = True
            return True

        return False

    def _reclaim(self, server_pid: int) -> None:
        """
        Kill an orphaned llama-server left behind by a crashed process
        and wait until the port is released.
        """
        self._log(
            f"Reclaiming orphaned llama-server (pid {server_pid})."
        )
        try:
            os.kill(server_pid, signal.SIGTERM)
        except OSError:
            pass
        self._pidfile.unlink(missing_ok=True)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not self._server_healthy():
                return
            time.sleep(0.5)
        try:
            os.kill(server_pid, signal.SIGKILL)
        except OSError:
            pass

    def stop(self) -> None:
        if self._adopted:
            # A server started by another process/instance must not be
            # killed here; its owner is responsible for stopping it.
            self._adopted = False
            return
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        self._remove_pid_file()

    def _download(
        self,
        url: str,
        dest: Path,
        label: str,
        progress_range: tuple[float, float] = (0.0, 100.0),
    ) -> None:
        tmp = dest.with_name(dest.name + ".part")
        start, end = progress_range

        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            total = int(response.headers.get("Content-Length") or 0)

            columns = [
                TextColumn(label, style="cyan"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
            ]
            if total:
                columns.append(TotalFileSizeColumn())
            columns.append(TimeRemainingColumn())

            downloaded = 0

            with Progress(*columns, console=self.console) as progress:
                task = progress.add_task("", total=total)
                with open(tmp, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                            downloaded += len(chunk)
                            progress.update(task, advance=len(chunk))
                            if total:
                                pct = start + (end - start) * (
                                    downloaded / total
                                )
                                self._report_progress(pct)

        self._report_progress(end)
        tmp.rename(dest)

    @staticmethod
    def _extract(archive: Path, dest_dir: Path) -> None:
        dest_resolved = dest_dir.resolve()
        if str(archive).endswith(".tar.gz"):
            with tarfile.open(archive, "r:gz") as tf:
                try:
                    tf.extractall(dest_dir, filter="data")
                except TypeError:
                    # Python < 3.12: без filter — проверяем вручную.
                    for member in tf.getmembers():
                        target = (dest_resolved / member.name).resolve()
                        if not target.is_relative_to(dest_resolved):
                            raise RuntimeError(
                                f"Unsafe path in archive: {member.name}"
                            )
                    tf.extractall(dest_dir)
        else:
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    target = (dest_resolved / info.filename).resolve()
                    if not target.is_relative_to(dest_resolved):
                        raise RuntimeError(
                            f"Unsafe path in archive: {info.filename}"
                        )
                zf.extractall(dest_dir)


#
# Shared server manager.
#
# Long-running processes (e.g. the Web UI) should reuse one server
# instance so the GGUF model is downloaded/loaded only once.
#
_shared_server: LocalLLMServer | None = None
_shared_lock = threading.Lock()


def get_shared_server(
    console: Console | None = None,
    log: Callable[[str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> LocalLLMServer:
    """
    Return the process-wide llama.cpp server, downloading/starting it
    on first use. Callers should not stop it.
    """
    global _shared_server
    with _shared_lock:
        if _shared_server is None:
            server = LocalLLMServer(
                console=console,
                log=log,
                on_progress=on_progress,
            )
            try:
                server.prepare_and_start()
            except Exception:
                server.stop()
                raise
            _shared_server = server
        return _shared_server


def local_model_status() -> dict:
    server = _shared_server
    if server is not None:
        data = server.selection.as_dict()
        memory = detect_memory()
        data.update(loaded=True, fits=True,
                    available_gib=round(memory.available / GIB, 2) if memory else None,
                    reason='Локальная модель уже загружена и используется.')
        return data
    return {**configured_selection().as_dict(), 'loaded': False}


def stop_shared_server() -> bool:
    """
    Stop and release the shared llama.cpp server (frees memory and the port).

    Returns True if a server was actually stopped, False otherwise.
    """
    global _shared_server
    with _shared_lock:
        if _shared_server is not None:
            _shared_server.stop()
            _shared_server = None
            return True
        return False
