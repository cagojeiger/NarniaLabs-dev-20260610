"""학습 잡의 관측 신호(로그 + 메트릭)를 표준 포맷으로 떨구는 모듈.

이 모듈이 만들어내는 두 산출물이 곧 **Python(학습) → Rust(진단 에이전트)** 의
유일한 인터페이스(계약)다. 에이전트는 코드를 호출하지 않고 아래 두 파일만 읽는다:

  <run_dir>/train.log     사람/에이전트가 읽는 텍스트 로그 (실제 프레임워크 스타일)
  <run_dir>/metrics.json  구조화된 메트릭/설정/환경/에러 (schema_version 고정)

설계 의도
---------
- 텍스트 로그와 구조화 메트릭을 **둘 다** 남긴다. 실제 학습 잡이 그렇기 때문이고,
  에이전트가 (a) 비정형 로그 파싱과 (b) 구조화 신호 활용을 모두 시연할 수 있어야 한다.
- JSON 에는 NaN/Inf 가 없으므로 비정상 수치는 null 로 저장하고, 별도의
  불리언 신호(`diverged`)와 텍스트 로그의 "loss=nan" 라인으로 명시한다.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_number(x: float | int | None) -> float | int | None:
    """JSON 안전 변환: NaN/Inf 는 null 로. (정상 수치는 그대로)"""
    if x is None:
        return None
    if isinstance(x, (int,)):
        return x
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


@dataclass
class StepRecord:
    """학습 한 스텝/에폭의 스칼라 신호."""

    step: int
    epoch: int
    train_loss: float | None = None
    val_loss: float | None = None
    train_acc: float | None = None
    val_acc: float | None = None
    grad_norm: float | None = None
    lr: float | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("train_loss", "val_loss", "train_acc", "val_acc", "grad_norm", "lr"):
            d[k] = _json_number(d[k])
        return d


@dataclass
class ErrorInfo:
    type: str
    message: str
    phase: str  # 예: data_prep | model_build | training_step | validation
    traceback_tail: list[str] = field(default_factory=list)


class RunRecorder:
    """하나의 학습 실행을 기록한다. with-블록으로 쓰면 종료 상태를 자동 확정한다."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        stage: str,
        config: dict[str, Any],
        framework: str = "pytorch",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "train.log"
        self.metrics_path = self.run_dir / "metrics.json"

        self.run_id = run_id
        self.stage = stage
        self.config = config
        self.framework = framework

        self.history: list[StepRecord] = []
        self.error: ErrorInfo | None = None
        self.status = "running"
        self._t0 = time.monotonic()
        self._started_at = _utcnow_iso()

        # 텍스트 로그 파일을 연다 (라인 단위 flush 로 중단돼도 흔적이 남게)
        self._log_fp = self.log_path.open("w", encoding="utf-8")
        self._env = _collect_env(framework)
        self.log(
            f"run start id={run_id} stage={stage} framework={framework} "
            f"device={self._env.get('device')} cuda={self._env.get('cuda_available')}"
        )
        self.log(f"config {json.dumps(config, ensure_ascii=False)}")

    # ---- 텍스트 로그 ----
    def log(self, msg: str, level: str = "INFO") -> None:
        line = f"{_utcnow_iso()} [{level}] {msg}"
        self._log_fp.write(line + "\n")
        self._log_fp.flush()
        # 콘솔에도 흘려 보내 dvc/도커 로그에 같이 남게 한다
        print(line, file=sys.stderr if level in ("ERROR", "WARNING") else sys.stdout)

    def warn(self, msg: str) -> None:
        self.log(msg, level="WARNING")

    # ---- 스텝 메트릭 ----
    def log_step(self, rec: StepRecord) -> None:
        self.history.append(rec)
        parts = [f"epoch={rec.epoch}", f"step={rec.step}"]
        for k in ("train_loss", "val_loss", "train_acc", "val_acc", "grad_norm", "lr"):
            v = getattr(rec, k)
            if v is not None:
                parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        self.log(" ".join(parts), level="INFO")

    # ---- 종료 처리 ----
    def fail(self, exc: BaseException, *, phase: str) -> None:
        tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tail = [ln.rstrip("\n") for ln in "".join(tb).splitlines()[-12:]]
        self.error = ErrorInfo(
            type=type(exc).__name__,
            message=str(exc),
            phase=phase,
            traceback_tail=tail,
        )
        self.status = "failed"
        self.log(f"{type(exc).__name__}: {exc}", level="ERROR")
        for ln in tail:
            self.log(ln, level="ERROR")

    def _finalize(self) -> None:
        if self.status == "running":
            self.status = "succeeded"
        duration = time.monotonic() - self._t0
        final = self.history[-1].to_json() if self.history else {}
        diverged = any(
            (r.train_loss is not None and not math.isfinite(r.train_loss))
            or (r.train_loss is None and r.step > 0)
            for r in self.history
        )
        doc = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "stage": self.stage,
            "framework": self.framework,
            "status": self.status,
            "started_at": self._started_at,
            "ended_at": _utcnow_iso(),
            "duration_sec": round(duration, 3),
            "config": self.config,
            "env": self._env,
            "diverged": diverged,
            "history": [r.to_json() for r in self.history],
            "final": final,
            "error": asdict(self.error) if self.error else None,
        }
        self.metrics_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.log(
            f"run end status={self.status} duration_sec={duration:.2f} "
            f"steps={len(self.history)} -> {self.metrics_path}"
        )
        self._log_fp.close()

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # 예외가 났는데 아직 기록 안 됐으면 여기서 기록한다.
        if exc is not None and self.error is None:
            self.fail(exc, phase="unknown")
        self._finalize()
        # 실패 시나리오에서도 프로세스 종료코드로 실패를 알리고 싶으므로
        # 예외는 삼키지 않는다 (dvc/워크플로우가 실패를 전파하도록).
        return False


def _collect_env(framework: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "cuda_available": False,
    }
    if framework == "pytorch":
        try:
            import torch

            env["torch"] = torch.__version__
            env["cuda_available"] = bool(torch.cuda.is_available())
            env["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # torch 가 없어도 telemetry 자체는 동작
            env["torch"] = None
    return env
