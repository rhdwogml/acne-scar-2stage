# acne-scar-2stage

얼굴 사진에서 여드름 흉터를 찾아 유형별 확률을 내는 2단계 추론 파이프라인입니다.

```
사진 ──▶ 1단계 YOLO11n (병변 위치 탐지) ──▶ 상자 크롭 ──▶ 2단계 분류 모델 (유형별 확률)
```

- 1단계: YOLO11n, 입력 640 × 640
- 2단계: 크롭을 224 × 224로 조정해 5개 유형으로 분류
  - `resnet50` (개선판): 모델 비교 실험에서 가장 높았던 구조. `weights/resnet50_scar.pt`가 있으면 자동으로 사용
  - `vit_cbam` (기존): ViT-B/16 + CBAM
- 유형: Boxcar, Ice pick, Rolling (이상 패인 흉터), Hypertrophic/Keloid, Normal

## 설치

가중치는 Git LFS로 올라가 있습니다. Git LFS가 설치된 상태에서 받으세요.

```
git lfs install
git clone https://github.com/rhdwogml/acne-scar-2stage.git
cd acne-scar-2stage
pip install -r requirements.txt
```

## 사용

명령줄:

```
python predict.py face.jpg --out result.jpg
```

파이썬(웹 서버에서 부를 때):

```python
from scar_pipeline import ScarPipeline

pipe = ScarPipeline()                 # 서버가 뜰 때 한 번만 만든다 (모델 로딩)
result = pipe.predict("face.jpg")     # 파일 경로 또는 PIL.Image
```

`result`는 그대로 JSON으로 내보낼 수 있는 dict입니다.

```json
{
  "image_size": [641, 356],
  "conf_threshold": 0.25,
  "stage2_model": "vit_cbam",
  "lesions": [
    {
      "box": [536, 170, 562, 186],
      "det_conf": 0.6314,
      "det_class": "Ice pick",
      "class": "Ice pick",
      "class_ko": "아이스픽",
      "prob": 0.6375,
      "probs": {"Boxcar": 0.36, "Hypertrophic/Keloid": 0.0014, "Normal": 0.0, "Rolling": 0.001, "Ice pick": 0.6375},
      "atrophic_prob": 0.9986
    }
  ],
  "summary": {"n_lesions": 82, "counts": {"Ice pick": 69, "Boxcar": 13}, "n_atrophic": 82, "mean_atrophic_prob": 0.9999}
}
```

| 값 | 뜻 |
|---|---|
| `box` | 병변 상자 `[x1, y1, x2, y2]` (원본 사진의 픽셀 좌표) |
| `det_conf`, `det_class` | 1단계가 낸 탐지 신뢰도와 유형 |
| `class`, `class_ko`, `prob` | 2단계가 고른 유형과 그 확률 |
| `probs` | 2단계의 유형별 확률 (합 1) |
| `atrophic_prob` | 패인 흉터일 확률 = Boxcar + Ice pick + Rolling 확률의 합 |
| `summary` | 사진 전체 요약 (유형별 개수, 패인 흉터 개수와 평균 확률) |

탐지가 너무 적으면 `pipe.predict(img, conf=0.15)`처럼 임계값을 낮출 수 있습니다. 낮출수록 오탐도 늘어납니다.

## 2단계 개선판(ResNet50) 가중치 만들기

모델 비교 실험에서는 학습이 끝난 가중치를 남기지 않았기 때문에, 한 번 다시 학습해야 합니다. GPU가 있는 PC에서 약 30분 걸립니다.

```
python train_stage2.py --data C:\path\to\scar_final_cls --model resnet50 --seed 2
git add weights/resnet50_scar.pt weights/resnet50_scar.json
git commit -m "Add ResNet50 stage-2 weights"
git push
```

`weights/resnet50_scar.pt`가 생기면 `ScarPipeline()`이 자동으로 ResNet50을 씁니다. 기존 모델로 돌리려면 `ScarPipeline(stage2="vit_cbam")`.

## 성능 (시험 집합, 영상 111장)

| 항목 | 값 |
|---|---|
| 1단계 YOLO11n mAP50 | 검증 0.577, 시험 0.507 |
| 2단계 ViT-CBAM macro-F1 (정답 상자 크롭 610개) | 0.705 (이 저장소의 가중치), 시드 반복 0.723 ± 0.005 |
| 2단계 ResNet50 macro-F1 (같은 크롭) | 0.773 ± 0.020 (시드 0·1·2: 0.762, 0.761, 0.796) |
| 패인 흉터 3종을 하나로 묶었을 때 macro-F1 | 0.839 |
| 통합(1단계 → 2단계)에서 탐지된 병변 비율 | 33.4% |
| 통합 macro-F1 | YOLO11n 단독 0.549, YOLO11n → ViT-CBAM 0.495 |

## 알아 둘 한계

- **놓치는 병변이 많습니다.** 통합 평가에서 실제 병변의 약 3분의 1만 탐지되었습니다. 결과에 없는 병변이 "없다"는 뜻은 아닙니다.
- **Boxcar와 Ice pick은 서로 자주 바뀝니다.** 두 유형의 F1은 0.47–0.66 수준입니다. 패인 흉터인지 여부(`atrophic_prob`)가 세부 유형(`class`)보다 믿을 만합니다.
- **통합 조건의 성능은 2단계 단독 성능보다 낮습니다.** 2단계는 정답 상자 크롭으로만 학습되었습니다. ResNet50을 2단계에 넣은 통합 성능은 아직 측정하지 않았습니다.
- **정상 피부 판별은 검증되지 않았습니다.** Normal 클래스는 얼굴 전체 사진으로 학습되어, 작은 피부 조각을 정상으로 가려내지 못합니다.
- **전문의가 검증한 라벨이 아닙니다.** 공개 데이터셋의 주석과 직접 단 주석을 합친 데이터로 학습했습니다.
- 이 모델은 연구·교육용이며 의료 진단을 대신하지 않습니다. 치료 안내를 붙일 때는 이 점을 화면에 밝혀야 합니다.

## 구성

```
scar_pipeline/model.py      2단계 모델 정의 (ViT-CBAM, ResNet50)
scar_pipeline/pipeline.py   ScarPipeline (탐지 → 크롭 → 분류)
predict.py                  명령줄 추론
train_stage2.py             2단계 모델 학습
weights/yolo11n_scar.pt     1단계 가중치 (5 MB)
weights/vit_cbam_scar.pt    2단계 ViT-CBAM 가중치 (328 MB)
```

학습 데이터와 얼굴 사진은 이 저장소에 넣지 않았습니다.
