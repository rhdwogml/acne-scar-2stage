# 팀 계약 (contract)

네 사람이 주고받는 데이터의 모양을 정해 둔 폴더입니다. 각자 LLM(Claude, Codex 등)으로 작업할 때 **이 폴더를 먼저 보여 주고 시작하세요.** 그러면 다른 사람 코드가 아직 없어도 가짜 JSON으로 먼저 만들고, 나중에 그대로 끼워 맞출 수 있습니다.

```
[공재희] 브라우저 추론 ──InferenceResult──▶ [정희재] 추천 엔진 ──Recommendation──▶ [김근호·최현정] 화면
                                             ▲ treatments.json, prices.json             │
                                             │                                          ▼
                                        [정희재] /api/explain ◀──ExplainRequest─── 화면
```

## 파일

| 파일 | 내용 | 누가 쓰나 |
|---|---|---|
| `types.ts` | 모든 데이터 모양과 함수 모양 (계약 본문) | 전원 |
| `check.ts` | JSON이 계약을 지키는지 검사 | 전원 |
| `data/treatments.json` | 유형별 시술 매핑표, 표준 시술 사전, 참고문헌 (제안서 3.2.5 그대로, **실제 데이터**) | 정희재 관리 |
| `mocks/inference.*.json` | 가짜 추론 결과 5종 | 화면, 추천 엔진 개발용 |
| `mocks/prices.mock.json` | 가짜 가격 (병원 이름·가격 모두 지어낸 값) | 추천 엔진, 화면 개발용 |
| `mocks/recommendation.*.json` | 추천 결과 예시 6개 (실제 엔진 출력) | 화면 개발용 |
| `mocks/explain.response.json` | 가짜 쉬운 설명 응답 | 화면 개발용 |

가짜 추론 결과 5종:

| 파일 | 상황 | 화면에서 확인할 것 |
|---|---|---|
| `inference.mixed_atrophic.json` | Ice pick 6, Boxcar 3, Rolling 2 (그중 2개는 Boxcar·Ice pick 모호) | 여러 유형, "Boxcar 또는 Ice Pick" 묶음, 공통 시술 |
| `inference.hypertrophic.json` | Hypertrophic 4, 확률 낮은 Rolling 1 | 솟은 흉터 추천, θ 미만 유형 제외 |
| `inference.normal_only.json` | Normal 2 | 추천 생략 안내 |
| `inference.no_lesion.json` | 흉터 0개 | "찾지 못함" 안내 (흉터가 없다는 뜻은 아님) |
| `inference.below_threshold.json` | 흉터 3개, 모두 확률 40% 미만 | 판정 보류 안내 |

## 사람별로 지킬 것

**공재희 (브라우저 추론)**
- 출력은 `InferenceResult`입니다. Python `ScarPipeline.predict()`의 JSON과 **키 이름·철자·대소문자까지 같게** 합니다. 특히 `class` 값은 `"Ice pick"`(p 소문자), `"Hypertrophic/Keloid"`입니다.
- `box`는 640으로 줄인 좌표가 아니라 **원본 사진 픽셀 좌표**입니다.
- 브라우저에서만 `timing_ms`를 채웁니다(13주차 시간 측정용).
- 같은 사진을 Python과 브라우저에서 돌려 결과를 비교하면 ONNX 정합성 검증이 됩니다.

**정희재 (추천 엔진, 가격, Gemini)**
- 함수 모양: `recommend(result: InferenceResult, region: Region, opts?) => Recommendation`
- 지역 태그를 누르면 화면이 이 함수를 다시 부릅니다. 그러니 빠르고 순수한 함수(같은 입력 → 같은 출력)로 만듭니다.
- 구현은 [`recommend/`](../recommend/README.md)에 있습니다. 제안서에 없어서 정한 해석 5가지도 거기 적어 두었습니다.
- `mocks/recommendation.*.json`은 실제 엔진이 만든 출력입니다(`node recommend/make-mocks.ts`). 엔진 규칙을 바꾸면 이 명령으로 다시 만듭니다.

**김근호·최현정 (화면)**
- 화면은 `Recommendation`만 보고 그립니다. 흉터 박스 그리기만 `InferenceResult.lesions[].box`를 씁니다.
- `status`별로 화면이 4가지입니다: `ok`, `normal_only`, `no_lesion`, `below_threshold`.
- 가격은 `price.status`로 나눕니다: `ok`는 평균가, `insufficient`는 "표본 부족"과 `range`, `no_data`는 "가격 정보 없음". `promotion_min`이 있으면 평균과 따로 표시하고, `collected_month`를 기준 월로 표시합니다.
- `disclaimer`와 근거 문헌 번호(`refs`)는 모든 추천 화면에 표시합니다(제안서 원칙).

**전원 (가격 수집)**
- `PriceEntry` 모양 그대로 적습니다. `price`는 쉼표·"원" 없는 정수, `is_promotion`은 `true`/`false`, `collected_month`는 `"2026-11"` 형식입니다.
- `treatment_raw`에는 병원 표기를 그대로 적고, `treatment`에는 `treatments.json`의 id를 적습니다. 어떤 id인지 모르면 `aliases`를 찾아보고, 없으면 정희재에게 알려 주세요(사전에 추가).
- 평균에는 정가이면서 `price_unit`이 `"1회"`인 가격만 들어갑니다. 패키지 가격도 적어 두되 단위를 정확히 적어 주세요.

## 검사하기

Node 22.18 이상이면 설치 없이 바로 됩니다.

```
node contract/check.ts                                  # 이 폴더의 예시 파일 전부
node contract/check.ts inference 내_출력.json            # 내 추론 결과 검사
node contract/check.ts prices data/prices.json           # 수집한 가격 파일 검사
```

틀린 곳이 있으면 어느 항목이 왜 틀렸는지 한국어로 알려 줍니다. 코드 안에서는 `import { checkInference } from "./check.ts"`처럼 불러 쓸 수 있습니다.

## Next.js 저장소가 생기면

이 폴더를 통째로 `web/lib/contract/`로 옮기고, 이 폴더 원본은 지웁니다. 복사본이 두 개가 되면 서로 달라집니다. `data/treatments.json`은 `web/data/`로 옮겨도 됩니다.

## 계약을 바꿀 때

혼자 고치지 말고 팀 채팅에 먼저 알립니다. 고친 뒤 `node contract/check.ts`가 통과하는지 확인하고, 커밋 메시지 앞에 `[contract]`를 붙입니다.
