/**
 * 계약 검사기: JSON이 types.ts 모양을 지키는지 확인한다.
 *
 *   node contract/check.ts                      → contract/ 안의 예시 파일 전부 검사
 *   node contract/check.ts inference my.json    → 내 출력 파일 하나 검사 (inference | recommendation | prices | mapping)
 *
 * Node 22.18 이상이면 설치 없이 .ts를 바로 실행한다. 브라우저 코드에서도 import 해서 쓸 수 있다.
 */
import type { InferenceResult, PriceFile, Recommendation, TreatmentMapping } from "./types.ts";

const CLASSES = ["Boxcar", "Ice pick", "Rolling", "Hypertrophic/Keloid", "Normal"];
const TYPES = CLASSES.filter((c) => c !== "Normal");
const REGIONS = ["천안", "아산", "수원", "강남", "용인"];
const TREATMENTS = [
  "tca_cross", "punch_excision", "rf", "subcision", "filler", "nonablative_laser",
  "laser_resurfacing", "tca_peel", "steroid_injection", "pdl", "pdl_steroid_5fu",
];

type Errors = string[];
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const is01 = (v: unknown) => isNum(v) && v >= 0 && v <= 1;
const isMonth = (v: unknown) => typeof v === "string" && /^\d{4}-(0[1-9]|1[0-2])$/.test(v);

function need(e: Errors, ok: boolean, msg: string) {
  if (!ok) e.push(msg);
}

export function checkInference(r: any): Errors {
  const e: Errors = [];
  need(e, Array.isArray(r?.image_size) && r.image_size.length === 2 && r.image_size.every(isNum), "image_size는 [가로, 세로] 숫자 2개");
  need(e, is01(r?.conf_threshold), "conf_threshold는 0~1 숫자");
  need(e, r?.stage2_model === "vit_cbam" || r?.stage2_model === "resnet50", "stage2_model은 vit_cbam 또는 resnet50");
  if (!Array.isArray(r?.lesions)) return [...e, "lesions는 배열"];
  const [W, H] = r.image_size ?? [Infinity, Infinity];
  r.lesions.forEach((l: any, i: number) => {
    const at = `lesions[${i}]`;
    const b = l?.box;
    need(e, Array.isArray(b) && b.length === 4 && b.every(isNum), `${at}.box는 [x1,y1,x2,y2]`);
    if (Array.isArray(b)) need(e, b[0] < b[2] && b[1] < b[3] && b[0] >= 0 && b[1] >= 0 && b[2] <= W && b[3] <= H, `${at}.box가 사진 밖이거나 뒤집힘 (원본 픽셀 좌표여야 함)`);
    need(e, is01(l?.det_conf), `${at}.det_conf는 0~1`);
    need(e, typeof l?.det_class === "string", `${at}.det_class는 문자열`);
    need(e, CLASSES.includes(l?.class), `${at}.class "${l?.class}"는 ${CLASSES.join(" / ")} 중 하나 (철자·대소문자 주의)`);
    need(e, typeof l?.class_ko === "string", `${at}.class_ko는 문자열`);
    const p = l?.probs ?? {};
    const keys = Object.keys(p);
    need(e, keys.length === 5 && CLASSES.every((c) => is01(p[c])), `${at}.probs는 5개 유형 모두 0~1 확률`);
    const sum = CLASSES.reduce((s, c) => s + (p[c] ?? 0), 0);
    need(e, Math.abs(sum - 1) < 0.01, `${at}.probs 합이 ${sum.toFixed(4)} (1이어야 함, softmax 확인)`);
    const top = CLASSES.reduce((a, c) => ((p[c] ?? -1) > (p[a] ?? -1) ? c : a), CLASSES[0]);
    need(e, l?.class === top, `${at}.class는 probs에서 가장 큰 유형(${top})이어야 함`);
    need(e, isNum(l?.prob) && Math.abs(l.prob - (p[l?.class] ?? -9)) < 1e-3, `${at}.prob는 probs[class]와 같아야 함`);
    const atro = (p.Boxcar ?? 0) + (p["Ice pick"] ?? 0) + (p.Rolling ?? 0);
    need(e, isNum(l?.atrophic_prob) && Math.abs(l.atrophic_prob - atro) < 2e-3, `${at}.atrophic_prob는 Boxcar+Ice pick+Rolling`);
  });
  const s = r?.summary;
  need(e, s?.n_lesions === r.lesions.length, "summary.n_lesions는 lesions 개수");
  const counts: Record<string, number> = {};
  r.lesions.forEach((l: any) => (counts[l.class] = (counts[l.class] ?? 0) + 1));
  need(e, CLASSES.every((c) => (s?.counts?.[c] ?? 0) === (counts[c] ?? 0)), "summary.counts가 lesions와 다름");
  const atroL = r.lesions.filter((l: any) => ["Boxcar", "Ice pick", "Rolling"].includes(l.class));
  need(e, s?.n_atrophic === atroL.length, "summary.n_atrophic이 lesions와 다름");
  need(e, is01(s?.mean_atrophic_prob), "summary.mean_atrophic_prob는 0~1");
  if (r?.timing_ms !== undefined) {
    const t = r.timing_ms;
    need(e, ["model_load", "detect", "classify", "total"].every((k) => isNum(t?.[k]) && t[k] >= 0), "timing_ms의 model_load·detect·classify·total은 0 이상 숫자(ms)");
  }
  return e;
}

function checkPriceStat(p: any, at: string, e: Errors) {
  need(e, REGIONS.includes(p?.region), `${at}.region`);
  need(e, TREATMENTS.includes(p?.treatment), `${at}.treatment`);
  need(e, ["ok", "insufficient", "no_data"].includes(p?.status), `${at}.status`);
  need(e, isNum(p?.n_clinics), `${at}.n_clinics`);
  if (p?.status === "ok") need(e, Number.isInteger(p?.average) && p.n_clinics >= 3, `${at}: ok면 average 정수, n_clinics ≥ 3`);
  if (p?.status === "insufficient") need(e, p?.average === null && p.n_clinics >= 1 && p.n_clinics < 3, `${at}: insufficient면 average null, n_clinics 1~2`);
  if (p?.status === "no_data") need(e, p?.average === null && p.n_clinics === 0 && p.range === null, `${at}: no_data면 average·range null, n_clinics 0`);
  need(e, p?.range === null || (Array.isArray(p?.range) && p.range.every(Number.isInteger) && p.range[0] <= p.range[1]), `${at}.range는 null 또는 [최저, 최고] 정수`);
  need(e, p?.promotion_min === null || Number.isInteger(p?.promotion_min), `${at}.promotion_min은 null 또는 정수`);
  need(e, p?.collected_month === null || isMonth(p?.collected_month), `${at}.collected_month는 YYYY-MM 또는 null`);
}

function checkRecTreatment(t: any, at: string, e: Errors) {
  need(e, TREATMENTS.includes(t?.treatment), `${at}.treatment "${t?.treatment}"는 표준 시술 id`);
  need(e, typeof t?.name === "string" && typeof t?.summary === "string" && typeof t?.evidence === "string", `${at}의 name·summary·evidence는 문자열`);
  need(e, TYPES.includes(t?.for_type), `${at}.for_type`);
  need(e, [1, 2, 3].includes(t?.rank), `${at}.rank는 1~3`);
  need(e, Array.isArray(t?.refs) && t.refs.every(Number.isInteger), `${at}.refs는 문헌 번호 배열`);
  checkPriceStat(t?.price, `${at}.price`, e);
}

export function checkRecommendation(r: any): Errors {
  const e: Errors = [];
  need(e, ["ok", "normal_only", "no_lesion", "below_threshold"].includes(r?.status), "status");
  need(e, is01(r?.params?.theta) && is01(r?.params?.delta), "params.theta·delta는 0~1");
  need(e, REGIONS.includes(r?.region), "region은 천안·아산·수원·강남·용인 중 하나");
  need(e, typeof r?.disclaimer === "string" && r.disclaimer.length > 0, "disclaimer 필수 (의료 진단 아님 안내)");
  need(e, isNum(r?.n_ambiguous_lesions), "n_ambiguous_lesions는 숫자");
  (r?.type_stats ?? []).forEach((s: any, i: number) => {
    need(e, CLASSES.includes(s?.type) && isNum(s?.count) && is01(s?.mean_prob) && typeof s?.selected === "boolean", `type_stats[${i}] 모양`);
    if (s?.type === "Normal") need(e, s.selected === false, `type_stats[${i}]: Normal은 selected false`);
  });
  const groups = r?.groups ?? [];
  if (r?.status !== "ok") need(e, groups.length === 0, `status가 ${r?.status}면 groups는 빈 배열`);
  if (r?.status === "ok") need(e, groups.length > 0, "status가 ok면 groups가 1개 이상");
  groups.forEach((g: any, i: number) => {
    const at = `groups[${i}]`;
    need(e, Array.isArray(g?.types) && g.types.length >= 1 && g.types.every((t: string) => TYPES.includes(t)), `${at}.types`);
    need(e, typeof g?.label === "string" && typeof g?.ambiguous === "boolean", `${at}.label·ambiguous`);
    if (g?.ambiguous) need(e, g.types.length === 2 && g.types.includes("Boxcar") && g.types.includes("Ice pick"), `${at}: 모호 묶음은 Boxcar·Ice pick 두 유형`);
    need(e, Array.isArray(g?.treatments) && g.treatments.length >= 1 && g.treatments.length <= 3 * g.types.length, `${at}.treatments는 유형당 최대 3개`);
    const ids = (g?.treatments ?? []).map((t: any) => t?.treatment);
    need(e, new Set(ids).size === ids.length, `${at}.treatments에 같은 시술이 두 번 있음`);
    (g?.treatments ?? []).forEach((t: any, j: number) => checkRecTreatment(t, `${at}.treatments[${j}]`, e));
    if (i > 0) need(e, groups[i - 1].count >= g.count, `groups는 count 많은 순이어야 함 (${at})`);
  });
  (r?.common ?? []).forEach((c: any, i: number) => {
    need(e, Array.isArray(c?.types) && c.types.length >= 2, `common[${i}].types는 2개 이상`);
    (c?.treatments ?? []).forEach((t: any, j: number) => checkRecTreatment(t, `common[${i}].treatments[${j}]`, e));
  });
  return e;
}

export function checkPrices(f: any): Errors {
  const e: Errors = [];
  need(e, typeof f?.updated === "string", "updated (YYYY-MM-DD)");
  if (!Array.isArray(f?.entries)) return [...e, "entries는 배열"];
  f.entries.forEach((p: any, i: number) => {
    const at = `entries[${i}]`;
    need(e, REGIONS.includes(p?.region), `${at}.region "${p?.region}"`);
    need(e, typeof p?.clinic === "string" && p.clinic.length > 0, `${at}.clinic`);
    need(e, typeof p?.treatment_raw === "string", `${at}.treatment_raw`);
    need(e, TREATMENTS.includes(p?.treatment), `${at}.treatment "${p?.treatment}"는 표준 시술 id`);
    need(e, Number.isInteger(p?.price) && p.price > 0, `${at}.price는 원 단위 정수 (쉼표·"원" 빼기)`);
    need(e, typeof p?.price_unit === "string", `${at}.price_unit`);
    need(e, typeof p?.is_promotion === "boolean", `${at}.is_promotion은 true/false`);
    need(e, isMonth(p?.collected_month), `${at}.collected_month는 YYYY-MM`);
    need(e, typeof p?.source === "string", `${at}.source`);
  });
  return e;
}

export function checkMapping(m: any): Errors {
  const e: Errors = [];
  need(e, TREATMENTS.every((t) => typeof m?.treatments?.[t]?.name === "string"), "treatments에 표준 시술 11개가 모두 있어야 함");
  TYPES.forEach((t) => {
    const list = m?.by_type?.[t];
    need(e, Array.isArray(list) && list.length === 3, `by_type["${t}"]는 3개`);
    (list ?? []).forEach((r: any, i: number) => {
      need(e, r?.rank === i + 1, `by_type["${t}"][${i}].rank 순서`);
      need(e, TREATMENTS.includes(r?.treatment), `by_type["${t}"][${i}].treatment`);
    });
  });
  (m?.common ?? []).forEach((c: any, i: number) => need(e, c?.treatments?.every((t: string) => TREATMENTS.includes(t)), `common[${i}].treatments`));
  return e;
}

// 타입 이름과 검사 함수 연결 (컴파일러가 types.ts와의 연결을 확인하도록)
export const checkers = {
  inference: checkInference as (v: InferenceResult | unknown) => Errors,
  recommendation: checkRecommendation as (v: Recommendation | unknown) => Errors,
  prices: checkPrices as (v: PriceFile | unknown) => Errors,
  mapping: checkMapping as (v: TreatmentMapping | unknown) => Errors,
};

// ───────── 명령줄 실행 ─────────
const isMain = typeof process !== "undefined" && process.argv[1]?.replace(/\\/g, "/").endsWith("contract/check.ts");
if (isMain) {
  const { readFileSync, readdirSync } = await import("node:fs");
  const { join, dirname } = await import("node:path");
  const { fileURLToPath } = await import("node:url");
  const here = dirname(fileURLToPath(import.meta.url));
  const [kind, file] = process.argv.slice(2);
  const jobs: [keyof typeof checkers, string][] = [];
  if (kind && file) {
    if (!(kind in checkers)) {
      console.error(`종류는 ${Object.keys(checkers).join(" | ")} 중 하나`);
      process.exit(2);
    }
    jobs.push([kind as keyof typeof checkers, file]);
  } else {
    jobs.push(["mapping", join(here, "data", "treatments.json")]);
    for (const f of readdirSync(join(here, "mocks"))) {
      const k = f.split(".")[0];
      if (k in checkers) jobs.push([k as keyof typeof checkers, join(here, "mocks", f)]);
    }
  }
  let bad = 0;
  for (const [k, f] of jobs) {
    const errs = checkers[k](JSON.parse(readFileSync(f, "utf8")));
    console.log(`${errs.length ? "✗" : "✓"} ${k.padEnd(14)} ${f.replace(/\\/g, "/").split("/").slice(-2).join("/")}`);
    errs.forEach((m) => console.log(`    - ${m}`));
    bad += errs.length ? 1 : 0;
  }
  process.exit(bad ? 1 : 0);
}
