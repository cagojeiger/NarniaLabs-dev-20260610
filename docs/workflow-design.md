# 워크플로우 시스템 설계 문서

> ML 학습 워크플로우 실패 진단 — (1) 설계 문서
> 본 문서는 (2) PoC(`dvc.yaml` + `mljob/`)와 함께 읽도록 작성되었으며, 모든 주장은 실제로 동작하는 코드를 가리킨다.

## Scope

**무엇을 만드는가.** ML 학습을 DAG(데이터 준비 → 학습 → 평가 → 배포)로 실행하고, 각 노드가 **표준화된 관측 신호(텍스트 로그 + 구조화 메트릭)**를 남기게 하는 최소 워크플로우 시스템. 이 신호가 (3) Rust 진단 에이전트의 입력이 된다.

**무엇을 만들지 않는가.** 분산 스케줄러, 멀티테넌시, 웹 콘솔 같은 플랫폼 기능은 범위 밖이다. Narnia의 현재 단계는 "단일 서버·로컬 중심의 빠른 실험"이므로, **새 실행 엔진을 직접 구현하는 대신 검증된 경량 엔진(DVC)을 채택**하고, 우리가 통제해야 하는 부분 — 즉 **관측 신호의 스키마** — 에 설계 역량을 집중했다.

**핵심 설계 결정 한 줄.** "워크플로우 엔진은 사지 말고 빌려라(DVC). 대신 Python 학습 잡과 Rust 진단 에이전트를 잇는 *로그/메트릭 계약*은 직접 설계하라."

## Architecture

```
                (2) 워크플로우 PoC                         (3) 진단 에이전트
   ┌─────────────────────────────────────────┐        ┌────────────────┐
   │  dvc.yaml  (DAG 정의: deps↔outs 매칭)      │        │   Rust agent    │
   │     │  dvc repro (topological 실행)        │        │                │
   │     ▼                                      │        │  reads only:    │
   │  data_prep → train → evaluate → deploy     │        │   train.log     │
   │     └ each: python -m mljob.<stage>        │        │   metrics.json  │
   │            │                               │        └────────▲───────┘
   │            ▼  telemetry.RunRecorder        │                 │
   │   runs/<stage>/{train.log, metrics.json}   │─────────────────┘
   └─────────────────────────────────────────┘   파일(JSON) = 유일한 계약
```

**컴포넌트 구성**

| 컴포넌트 | 역할 | 구현 |
|---|---|---|
| DAG 정의 | 노드·의존성을 YAML로 선언 | `dvc.yaml` (stages, deps, outs, metrics) |
| 실행 엔진 | 위상정렬 후 노드 순차 실행, content-hash 캐시 | DVC (`dvc repro`) |
| 노드 구현 | 실제 PyTorch 학습 잡 | `mljob/{data_prep,train,evaluate,deploy}.py` |
| 관측 신호 | 로그/메트릭을 표준 스키마로 기록 | `mljob/telemetry.py` (**우리가 설계한 계약**) |
| 실행 환경 | 의존성 격리·재현 | Docker 이미지 (로컬 의존성 0) |

**DAG 표현·실행 방식.** 노드는 `dvc.yaml`의 `stages`로 선언한다. DVC는 한 노드의 `outs`와 다른 노드의 `deps`가 같은 경로를 가리키면 **그 사이에 엣지를 자동 생성**한다 — 그래프를 수동으로 그릴 필요가 없다. 예: `train.outs = artifacts/model.pt`, `evaluate.deps ⊇ artifacts/model.pt` ⇒ `train → evaluate`. 실행은 `dvc repro`가 위상정렬 순서로 각 노드의 `cmd`를 서브프로세스로 돌린다. `dvc dag`로 그래프를 ASCII로 시각화한다(가산 항목).

**관측 신호 스키마(계약).** 모든 노드는 `telemetry.RunRecorder`를 통해 두 파일을 남긴다:
- `train.log` — 타임스탬프·레벨이 붙은 텍스트 로그(실제 프레임워크 스타일). 에이전트의 **비정형 로그 파싱** 능력을 시험.
- `metrics.json` — `schema_version`으로 고정된 구조화 신호: `config`(하이퍼파라미터), `env`(torch/디바이스), `history`(스텝별 loss·grad_norm·acc), `error`(타입·메시지·phase·traceback_tail), `diverged` 플래그. JSON에 없는 NaN/Inf는 `null` + `diverged:true`로 명시한다.

## Failure modes

**로그·메트릭 수집.** `RunRecorder`는 라인 단위 flush로 로그를 쓰므로 프로세스가 중간에 죽어도 마지막 상태가 남는다. `with` 블록 종료 시 `metrics.json`을 항상 확정 기록한다(성공·실패 무관). 실패 시 예외 타입·메시지·발생 phase·traceback 꼬리를 구조화해 저장한다.

**실패 전파(propagation).** `dvc repro`는 한 노드가 비정상 종료(exit≠0)하면 **거기서 멈추고 하위 노드를 실행하지 않는다** = fail-fast 전파. 노드 코드도 실패 시 종료코드 1을 반환하도록 했다(`return 1` + 예외 비-삼킴). 따라서 학습이 깨지면 evaluate/deploy는 자동으로 차단된다.

**재시도(retry) 정책.** DVC에는 노드 단위 자동 재시도가 없다. 대신 두 가지로 다룬다: (a) **재실행 = 이어하기.** `dvc repro`는 캐시(`dvc.lock`의 content-hash)를 보고 성공한 상위 노드를 건너뛰고 **실패한 노드부터 재개**한다 — 비싼 데이터 준비를 반복하지 않는다. (b) **결정적 실패는 재시도 무의미.** 본 PoC의 3개 실패(NaN, shape mismatch, overfitting)는 환경이 아니라 코드/설정에서 비롯한 결정적 실패라 재시도가 아니라 **수정**이 답이다. 일시적 실패(OOM, 네트워크)에 한해 노드 래퍼에서 `max_retries`+백오프를 둘 자리를 남겨뒀다(향후 작업).

**관측 신호 충분성.** 각 시나리오는 원인 추정 + 다음 액션을 끌어낼 만큼의 신호를 갖는다 — NaN: `history`의 loss=null/grad_norm 폭증 + `error.type=FloatingPointError` + `config.lr`; shape: `error`의 `RuntimeError: mat1 and mat2 ... (64x1568 and 784x128)`; overfitting: 크래시 없이 `history`에서 train_loss↓·val_loss↑ 갭. (상세는 README의 시나리오 표 참조.)

## Trade-offs vs Argo

| 축 | 본 설계 (DVC, 단일 서버) | Argo Workflows (K8s) |
|---|---|---|
| 전제 인프라 | 없음 — 로컬/단일 서버 바이너리 | Kubernetes 클러스터 + 컨트롤 플레인 |
| 셋업 비용 | `dvc init` 한 번 | 클러스터·CRD·RBAC 운영 |
| DAG 정의 | `dvc.yaml`(deps↔outs로 엣지 추론) | YAML(명시적 templates/steps) |
| 캐시/재현 | content-hash 캐시 + `dvc.lock` **기본 내장** | 별도 구성(아티팩트 repo, memoization) |
| 데이터 버전관리 | 1급 기능(데이터·모델 해시) | 범위 밖(외부 도구 필요) |
| 확장성 | 단일 노드 한계 | 수천 병렬 Pod로 수평 확장 |
| 적합 단계 | **빠른 실험·데모(현재 Narnia)** | 대규모 멀티팀 프로덕션 |

**요지.** Argo의 강점(대규모 병렬·멀티테넌시)은 현재 단계에서 필요 없는 복잡도다. DVC는 K8s 없이도 DAG·캐시·데이터 버전관리를 주며, "단일 서버 중심의 빠른 실험"이라는 회사 컨텍스트에 정확히 들어맞는다. 스케일이 커지면 같은 `mljob/` 노드 코드를 Argo 컨테이너로 그대로 감싸 이전할 수 있게 노드를 CLI 진입점으로 격리해 두었다(락인 회피).

## What I intentionally skipped

- **자체 실행 엔진 구현** — 과제가 "본격 엔진 불필요"라 명시했고, 시간을 핵심 산출물(3)에 쓰기 위해 검증된 DVC를 채택. 직접 만든다면 위상정렬·캐시·락파일을 재발명하게 된다.
- **K8s/원격 실행** — 현재 단계 불필요(선택 항목). 노드를 CLI로 격리해 이전 경로만 열어둠.
- **노드 자동 재시도·일시적 실패 처리** — 본 PoC의 실패는 결정적이라 우선순위 낮음. 훅 위치만 남김.
- **분산 데이터 로딩·GPU** — CPU에서 10분 내 재현을 최우선으로 둠(작은 FashionMNIST 서브셋).
- **human approval 게이트의 완전 구현** — 배포 전 승인은 (3) 에이전트 쪽 confidence threshold와 함께 다루는 것이 자연스러워 그쪽으로 미룸.
