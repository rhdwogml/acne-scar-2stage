/**
 * 패인킬러 팀 계약 파일 (v1, 2026-11-28)
 *
 * 네 사람이 주고받는 데이터 모양을 여기 한곳에 정해 둔다.
 * 각자 LLM에게 이 파일을 먼저 보여 주고, 이 모양대로만 입력을 받고 출력을 내게 한다.
 * 모양을 바꿔야 하면 혼자 고치지 말고 팀 채팅에 먼저 알린 뒤 이 파일을 고친다.
 *
 *   [공재희] 브라우저 추론  --InferenceResult-->  [정희재] 추천 엔진  --Recommendation-->  [김근호·최현정] 화면
 *                                                     ^ treatments.json, prices.json        |
 *                                                     |                                     v
 *                                                [정희재] /api/explain  <--ExplainRequest-- 화면
 *
 * 이름 규칙: Python predict()가 내는 JSON과 똑같이 맞추기 위해 키는 snake_case를 쓴다.
 */

/* ───────────────────────── 1. 흉터 유형 ───────────────────────── */

/** 2단계 분류 모델이 내는 5개 유형. 철자와 대소문자까지 pipeline.py의 LABELS와 같다. */
export type ScarClass = "Boxcar" | "Ice pick" | "Rolling" | "Hypertrophic/Keloid" | "Normal";

/** 추천 대상이 될 수 있는 유형 (Normal 제외) */
export type ScarType = Exclude<ScarClass, "Normal">;

export const SCAR_CLASSES: readonly ScarClass[] = ["Boxcar", "Ice pick", "Rolling", "Hypertrophic/Keloid", "Normal"];

export const SCAR_CLASS_KO: Record<ScarClass, string> = {
  Boxcar: "박스카",
  "Ice pick": "아이스픽",
  Rolling: "롤링",
  "Hypertrophic/Keloid": "비후성·켈로이드성",
  Normal: "정상 피부",
};

/** 유형별 확률. 다섯 값의 합은 1 (반올림 때문에 ±0.001 정도 어긋날 수 있음) */
export type ClassProbs = Record<ScarClass, number>;

/* ───────────────────────── 2. 추론 결과 (공재희 → 정희재·화면) ───────────────────────── */

/** 흉터 하나. Python ScarPipeline.predict()의 lesions[] 항목과 같다. */
export interface Lesion {
  /** [x1, y1, x2, y2], 원본 사진의 픽셀 좌표 (리사이즈 전 기준) */
  box: [number, number, number, number];
  /** 1단계 YOLO 탐지 신뢰도 0~1 */
  det_conf: number;
  /** 1단계 YOLO가 붙인 유형 이름 (참고용. 화면과 추천에는 class를 쓴다) */
  det_class: string;
  /** 2단계가 고른 유형 = probs에서 가장 큰 값 */
  class: ScarClass;
  class_ko: string;
  /** class의 확률 = probs[class] */
  prob: number;
  probs: ClassProbs;
  /** 패인 흉터일 확률 = Boxcar + Ice pick + Rolling */
  atrophic_prob: number;
}

export interface InferenceSummary {
  n_lesions: number;
  /** 유형별 흉터 개수 (0개인 유형은 키가 없을 수 있음) */
  counts: Partial<Record<ScarClass, number>>;
  /** class가 Boxcar·Ice pick·Rolling인 흉터 수 */
  n_atrophic: number;
  /** 위 흉터들의 atrophic_prob 평균, 없으면 0 */
  mean_atrophic_prob: number;
}

export interface InferenceResult {
  /** [가로, 세로] 원본 사진 크기 */
  image_size: [number, number];
  /** 1단계 탐지 임계값 (기본 0.25) */
  conf_threshold: number;
  stage2_model: "vit_cbam" | "resnet50";
  lesions: Lesion[];
  summary: InferenceSummary;
  /**
   * 브라우저 추론에서만 채우는 값 (Python은 내지 않음). 13주차 기기별 시간 측정용.
   * model_download는 첫 실행 때만 있고, 캐시에서 읽으면 0 또는 생략.
   */
  timing_ms?: {
    model_download?: number;
    model_load: number;
    detect: number;
    classify: number;
    total: number;
  };
}

/** 공재희가 만들 함수의 모양 */
export interface InferenceOptions {
  /** 1단계 탐지 임계값, 기본 0.25 */
  conf?: number;
  /** 첫 다운로드 진행률 등 화면에 보여 줄 진행 상황 */
  onProgress?: (p: { stage: "download" | "load" | "detect" | "classify"; loaded?: number; total?: number }) => void;
}
export type RunInference = (image: Blob | HTMLImageElement | ImageBitmap, opts?: InferenceOptions) => Promise<InferenceResult>;

/* ───────────────────────── 3. 매핑표 data/treatments.json (정희재 작성) ───────────────────────── */

/** 표준 시술 분류 id. 가격 데이터와 매핑표가 모두 이 id로 연결된다. */
export type TreatmentId =
  | "tca_cross"
  | "punch_excision"
  | "rf"
  | "subcision"
  | "filler"
  | "nonablative_laser"
  | "laser_resurfacing"
  | "tca_peel"
  | "steroid_injection"
  | "pdl"
  | "pdl_steroid_5fu";

export interface TreatmentDef {
  /** 화면에 보일 표준 이름, 예: "서브시전" */
  name: string;
  /** 일반 사용자용 한 줄 설명 (Gemini 실패 시 템플릿 문장에도 씀) */
  summary: string;
  /** 표준 시술 사전: 병원 표기 → 이 id로 통일. 예: ["서브시젼", "서브시전 주사"] */
  aliases: string[];
}

export interface TreatmentRank {
  rank: 1 | 2 | 3;
  treatment: TreatmentId;
  /** 이 유형에서 부르는 이름이 표준 이름과 다를 때만. 예: Boxcar의 "펀치 거상·절제" */
  label?: string;
  /** 근거 한 줄 (제안서 3.2.5 표) */
  evidence: string;
  /** 참고문헌 번호 (제안서 번호 그대로) */
  refs: number[];
}

export interface TreatmentMapping {
  version: string;
  treatments: Record<TreatmentId, TreatmentDef>;
  by_type: Record<ScarType, TreatmentRank[]>;
  /** 복합 흉터 공통 후보. 제안서 예시를 그대로 적는다(자동 교집합 아님). */
  common: { types: ScarType[]; treatments: TreatmentId[] }[];
  references: Record<string, string>;
}

/* ───────────────────────── 4. 가격 데이터 data/prices.json (전원 수집) ───────────────────────── */

export type Region = "천안" | "아산" | "수원" | "강남" | "용인";
export const REGIONS: readonly Region[] = ["천안", "아산", "수원", "강남", "용인"];

/** 가격 한 건 = 한 병원의 한 시술 가격 한 줄 */
export interface PriceEntry {
  region: Region;
  clinic: string;
  /** 병원 홈페이지에 적힌 원래 시술명 그대로 */
  treatment_raw: string;
  /** treatment_raw를 표준 시술 사전으로 바꾼 id */
  treatment: TreatmentId;
  /** 원 단위 숫자 (쉼표·"원" 없이) */
  price: number;
  /** 가격 기준. 평균에는 "1회"만 들어간다. 예: "1회", "3회 패키지", "1cc" */
  price_unit: string;
  is_promotion: boolean;
  /** "YYYY-MM" */
  collected_month: string;
  /** 캡처 파일 경로, 예: "captures/cheonan/2026-11_clinicA.png" */
  source: string;
}

export interface PriceFile {
  updated: string; // "YYYY-MM-DD"
  entries: PriceEntry[];
}

/** 한 지역·한 시술의 가격 요약 (추천 엔진이 계산해서 Recommendation 안에 넣음) */
export interface PriceStat {
  region: Region;
  treatment: TreatmentId;
  /**
   * ok: 정가(1회) 낸 병원이 3곳 이상 → average 표시
   * insufficient: 1~2곳 → "표본 부족" + range 표시
   * no_data: 0곳
   */
  status: "ok" | "insufficient" | "no_data";
  /** 병원별 정가(1회)의 평균, 원 단위 정수. status가 ok일 때만 숫자 */
  average: number | null;
  /** 정가를 낸 병원 수 */
  n_clinics: number;
  /** 정가 최저·최고. no_data면 null */
  range: [number, number] | null;
  /** 그 달 최저 프로모션 가격 (평균과 별도 표시). 없으면 null */
  promotion_min: number | null;
  /** 가격 기준 월 "YYYY-MM". no_data면 null */
  collected_month: string | null;
}

/* ───────────────────────── 5. 추천 결과 (정희재 → 화면) ───────────────────────── */

/** 유형별 집계 (규칙 ①). 화면의 "유형별 개수·확률" 표에 그대로 씀 */
export interface TypeStat {
  type: ScarClass;
  type_ko: string;
  count: number;
  /** 그 유형으로 분류된 흉터들의 prob 평균 */
  mean_prob: number;
  /** mean_prob ≥ θ 이고 Normal이 아니면 true (규칙 ②) */
  selected: boolean;
}

export interface RecommendedTreatment {
  treatment: TreatmentId;
  /** 화면에 쓸 이름 (label이 있으면 label, 없으면 표준 name) */
  name: string;
  summary: string;
  /** 어느 유형의 몇 순위로 들어왔는지 */
  for_type: ScarType;
  rank: 1 | 2 | 3;
  evidence: string;
  refs: number[];
  /** 선택한 지역의 가격 (규칙 ⑦) */
  price: PriceStat;
}

/** 화면에 한 덩어리로 보이는 유형 묶음 */
export interface RecommendationGroup {
  /** 보통 1개. Boxcar·Ice pick이 모호하면 ["Boxcar", "Ice pick"] (규칙 ③) */
  types: ScarType[];
  /** 화면 제목. 예: "Rolling", "Boxcar 또는 Ice Pick" */
  label: string;
  ambiguous: boolean;
  /** 이 묶음에 든 흉터 수 (정렬 기준, 규칙 ④) */
  count: number;
  mean_prob: number;
  /** 유형별 최대 3개 (규칙 ⑤). 모호 묶음이면 두 유형 것을 합쳐 최대 6개, 겹치면 한 번만 */
  treatments: RecommendedTreatment[];
}

export interface Recommendation {
  /**
   * ok: 추천 있음
   * normal_only: Normal만 검출 → 추천 생략 (규칙 ②)
   * no_lesion: 검출된 흉터 0개
   * below_threshold: 흉터는 있지만 θ를 넘는 유형이 없음
   */
  status: "ok" | "normal_only" | "no_lesion" | "below_threshold";
  params: { theta: number; delta: number };
  region: Region;
  /** count 많은 순 */
  type_stats: TypeStat[];
  /** count 많은 순 (규칙 ④). status가 ok가 아니면 빈 배열 */
  groups: RecommendationGroup[];
  /** 위축성 유형이 2개 이상일 때 공통 후보 (규칙 ⑥). 없으면 빈 배열 */
  common: { types: ScarType[]; treatments: RecommendedTreatment[] }[];
  /** Boxcar·Ice pick 확률 차이가 δ 미만이었던 흉터 수 */
  n_ambiguous_lesions: number;
  /** 모든 추천 화면에 표시 (제안서: 의료 진단 아님 안내) */
  disclaimer: string;
}

/** 정희재가 만들 함수의 모양. 지역 태그를 바꾸면 화면이 이 함수를 다시 부르면 된다(빠름). */
export type Recommend = (
  result: InferenceResult,
  region: Region,
  opts?: { theta?: number; delta?: number },
) => Recommendation;

export const DEFAULT_THETA = 0.4;
export const DEFAULT_DELTA = 0.15;

/* ───────────────────────── 6. 쉬운 설명 API  POST /api/explain ───────────────────────── */

/** 화면 → 서버리스 함수. 사진·박스 좌표는 보내지 않는다(제안서 개인정보 원칙). */
export interface ExplainRequest {
  recommendation: Recommendation;
}

export interface ExplainResponse {
  /** 사용자에게 보여 줄 쉬운 설명 문단 */
  text: string;
  /** gemini: 정상 생성 / template: 호출 실패·한도 초과·매핑표 밖 시술명 검출로 템플릿 대체 */
  source: "gemini" | "template";
}
