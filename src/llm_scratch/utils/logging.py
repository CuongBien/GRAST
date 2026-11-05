"""Isolated run directories, immutable config snapshots, and durable run lifecycle logs."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from llm_scratch.utils.config import LoadedConfig, canonical_json
from llm_scratch.utils.environment import collect_metadata
from llm_scratch.utils.io import write_json


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Run:
    def __init__(self, config: LoadedConfig):
        self.config = config
        data = config.data
        self.started_at = utc_now()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        name = (
            f"{stamp}-{data['experiment']['name']}-"
            f"seed{data['reproducibility']['seed']}-{uuid.uuid4().hex[:8]}"
        )

        self.path = Path(data["artifacts"]["root"]) / name
        self.path.mkdir(parents=True, exist_ok=False)
        for directory in ("logs", "checkpoints", "artifacts", "config_sources"):
            (self.path / directory).mkdir()
        self.logger = logging.getLogger(f"grast.run.{uuid.uuid4().hex}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.FileHandler(self.path / "logs" / "run.log", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def __enter__(self) -> Run:
        try:
            write_json(self.path / "config.json", self.config.data)
            for section, source in self.config.sources.items():
                (self.path / "config_sources" / f"{section}.toml").write_text(
                    source, encoding="utf-8"
                )
            self.metadata = collect_metadata(
                self.config.path,
                self.config.sha256,
                self.config.data["reproducibility"]["seed"],
            )
            write_json(self.path / "metadata.json", self.metadata)
            self._status("running")
            self.log("run_started", config_sha256=self.config.sha256)
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    def log(self, event: str, **fields: Any) -> None:
        record = {"timestamp": utc_now(), "event": event, **fields}
        with (self.path / "logs" / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(record) + "\n")
        self.logger.info("%s %s", event, canonical_json(fields))

    def _status(self, status: str, error: str | None = None) -> None:
        write_json(
            self.path / "run.json",
            {
                "status": status,
                "started_at": self.started_at,
                "finished_at": None if status == "running" else utc_now(),
                "error": error,
            },
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if exc is None:
                self.log("run_completed")
                self._status("completed")
            else:
                self.log("run_failed", error=str(exc), error_type=type(exc).__name__)
                self._status("failed", str(exc))
        finally:
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
