"""워크플로우 4번 노드: 배포 (mock).

실제 배포 대신, 학습된 모델을 배포 산출물 디렉터리로 복사하고
배포 매니페스트를 남긴다. PoC 의 DAG 마지막 노드를 채우는 용도.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from mljob import ARTIFACT_DIR
from mljob.telemetry import RunRecorder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/deploy")
    args = ap.parse_args()

    cfg = {"target": "local", "strategy": "copy-artifact"}
    with RunRecorder(args.run_dir, run_id="deploy", stage="deploy", config=cfg) as rec:
        src = Path(ARTIFACT_DIR) / "model.pt"
        dst_dir = Path(ARTIFACT_DIR) / "deployed"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "model.pt")

        manifest = {
            "model": "SmallCNN",
            "artifact": str(dst_dir / "model.pt"),
            "size_bytes": src.stat().st_size,
        }
        (dst_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rec.log(f"deployed -> {dst_dir} ({manifest['size_bytes']} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
