
from __future__ import annotations

import threading
import uuid
from dataclasses import fields
from typing import Callable

from ai_transcriber.utils.redaction import redact_diagnostic

from .models import Job


class JobManager:
    #
    # На MPS (Apple Silicon) память ограничена, поэтому задачи
    # транскрибации выполняются строго по одной: несколько
    # параллельных пайплайнов гарантированно исчерпают память.
    #
    _serial_lock = threading.Semaphore(1)

    MAX_JOBS = 20
    MAX_LOGS = 500

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, filename: str) -> Job:
        # Безопасное имя: только basename, без директорий.
        safe_filename = filename.split("/")[-1].split("\\")[-1][:200]
        job = Job(id=uuid.uuid4().hex, filename=safe_filename)
        with self._lock:
            if sum(value.status in ("queued", "running") for value in self._jobs.values()) >= self.MAX_JOBS:
                raise RuntimeError("Job queue is full; try again after a job finishes")
            self._jobs[job.id] = job
            # Eviction: не держим транскрипты в RAM вечно.
            completed = [key for key, value in self._jobs.items()
                         if value.status in ("completed", "failed")]
            for key in completed[:max(0, len(self._jobs) - self.MAX_JOBS)]:
                self._jobs.pop(key)
        return job

    def active_ids(self) -> set[str]:
        with self._lock:
            return {key for key, job in self._jobs.items()
                    if job.status in ("queued", "running")}

    def discard(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self, job_id: str) -> dict | None:
        job = self.get(job_id)
        if job is None:
            return None
        data = {field.name: getattr(job, field.name) for field in fields(job)
                if field.name not in ("transcript_text", "summary_text")}
        data["logs"] = list(job.logs)
        return data

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for key, value in kwargs.items():
                    setattr(job, key, value)

    def log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.log(message)

    def run(self, job_id: str, fn: Callable[[], None]) -> None:
        def worker():
            try:
                with JobManager._serial_lock:
                    self.update(job_id, status="running")
                    fn()
                self.update(
                    job_id,
                    status="completed",
                    progress=100,
                    current_step="Completed",
                )
            except Exception as exc:
                error = redact_diagnostic(exc)
                self.log(job_id, f"ERROR: {error}")
                self.update(
                    job_id,
                    status="failed",
                    current_step="Failed",
                    error=error,
                )

        threading.Thread(
            target=worker,
            name=f"ai-transcriber-job-{job_id}",
            daemon=True,
        ).start()
