"""Narnia ML 학습 잡 (워크플로우 PoC 의 각 노드 구현 + 실패 시나리오 생성기).

각 모듈은 `python -m mljob.<stage>` 로 단독 실행 가능하며,
표준 관측 신호(train.log + metrics.json)를 telemetry 모듈을 통해 남긴다.
"""

PROCESSED_DIR = "data/processed"
ARTIFACT_DIR = "artifacts"
