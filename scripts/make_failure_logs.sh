#!/usr/bin/env bash
# 3개의 재현 가능한 학습 실패 시나리오 로그를 생성한다.
# 각 실행은 scenarios/<fault>/ 에 train.log + metrics.json 을 남기며,
# 이 파일들이 (3) Rust 진단 에이전트의 입력이 된다. (scenarios/ 는 커밋되는 고정 입력본)
#
#   ./scripts/make_failure_logs.sh
#
# 사전조건: .data/processed 가 있어야 한다 (없으면 data_prep 먼저 실행).
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="uv run --frozen python -m"
[[ "${NO_UV:-0}" == "1" ]] && RUN="python -m"   # 컨테이너 등 uv 없는 환경용

if [[ ! -f .data/processed/train.pt ]]; then
  echo ">> .data/processed 없음 -> data_prep 먼저 실행"
  $RUN mljob.data_prep
fi

for fault in nan shape overfit; do
  echo ">> 실패 시나리오 생성: fault=${fault}"
  # 학습은 의도적으로 실패하므로(또는 진단대상 메트릭을 남기므로) 종료코드를 무시한다.
  $RUN mljob.train --fault "${fault}" || true
done

echo
echo ">> 완료. 생성된 로그:"
ls -1 scenarios/*/ | sed 's/^/   /'
