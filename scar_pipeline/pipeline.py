"""2단계 파이프라인: 1단계 YOLO11n 탐지 -> 상자 크롭 -> 2단계 분류(ResNet50 또는 ViT-CBAM)

    from scar_pipeline import ScarPipeline
    pipe = ScarPipeline()                 # 가중치는 weights/ 에서 읽는다
    result = pipe.predict("face.jpg")     # dict (JSON 으로 바로 내보낼 수 있음)

2단계 모델은 weights/resnet50_scar.pt 가 있으면 ResNet50(개선판), 없으면 ViT-CBAM(기존)을 쓴다.
ScarPipeline(stage2="vit_cbam") 처럼 직접 고를 수도 있다.
"""
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from .model import IMG_SIZE, build_stage2

ROOT = Path(__file__).resolve().parent.parent
YOLO_WEIGHT = ROOT / "weights" / "yolo11n_scar.pt"
STAGE2_WEIGHTS = {
    "resnet50": ROOT / "weights" / "resnet50_scar.pt",     # 개선판 (모델 비교 실험에서 가장 높았던 구조)
    "vit_cbam": ROOT / "weights" / "vit_cbam_scar.pt",     # 기존 2단계 모델
}

# 2단계 분류 모델의 출력 순서 (학습 시 ImageFolder 의 폴더명 정렬 순서)
VIT_CLASSES = ["Boxcar Scar", "Hypertrophic Scar", "Normal Skin", "Rolling Scar", "ice pick Scar"]
# 화면에 보여 줄 이름
LABELS = {
    "Boxcar Scar": "Boxcar",
    "ice pick Scar": "Ice pick",
    "Rolling Scar": "Rolling",
    "Hypertrophic Scar": "Hypertrophic/Keloid",
    "Normal Skin": "Normal",
}
LABELS_KO = {
    "Boxcar Scar": "박스카",
    "ice pick Scar": "아이스픽",
    "Rolling Scar": "롤링",
    "Hypertrophic Scar": "비후성·켈로이드성",
    "Normal Skin": "정상 피부",
}
# 패인(위축성) 흉터에 해당하는 클래스
ATROPHIC = ("Boxcar Scar", "ice pick Scar", "Rolling Scar")
COLORS = {"Boxcar": "#e6194b", "Ice pick": "#4363d8", "Rolling": "#3cb44b", "Hypertrophic/Keloid": "#f58231", "Normal": "#808080"}


def _check_weight(path):
    if not path.exists():
        raise FileNotFoundError(f"가중치 파일이 없습니다: {path}")
    if path.stat().st_size < 1024:      # Git LFS 포인터만 받아 온 경우
        raise RuntimeError(f"{path.name} 이 Git LFS 포인터 파일입니다. 저장소 폴더에서 'git lfs pull' 을 실행하세요.")


def _is_real(path):
    return path.exists() and path.stat().st_size >= 1024


class ScarPipeline:
    def __init__(self, stage2="auto", yolo_weight=YOLO_WEIGHT, stage2_weight=None, device=None):
        from ultralytics import YOLO

        if stage2 == "auto":
            stage2 = "resnet50" if _is_real(STAGE2_WEIGHTS["resnet50"]) else "vit_cbam"
        yolo_weight = Path(yolo_weight)
        stage2_weight = Path(stage2_weight) if stage2_weight else STAGE2_WEIGHTS[stage2]
        _check_weight(yolo_weight)
        _check_weight(stage2_weight)
        self.stage2 = stage2
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(str(yolo_weight))
        self.vit = build_stage2(stage2, num_classes=len(VIT_CLASSES))      # 2단계 분류 모델
        self.vit.load_state_dict(torch.load(str(stage2_weight), map_location="cpu", weights_only=True))
        self.vit.to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, image, conf=0.25, max_lesions=300):
        """image: 파일 경로 또는 PIL.Image.  conf: 1단계 탐지 신뢰도 임계값.

        반환값(dict)
          lesions: 탐지된 병변 목록. 각 항목에
            box            [x1, y1, x2, y2] 픽셀 좌표
            det_conf       1단계 탐지 신뢰도
            det_class      1단계(YOLO)가 예측한 유형
            class          2단계 분류 모델이 예측한 유형
            class_ko       위 유형의 한글 이름
            prob           2단계가 예측한 유형의 확률
            probs          유형별 확률 (합 1)
            atrophic_prob  패인 흉터(Boxcar + Ice pick + Rolling)일 확률
          summary: 영상 단위 요약 (유형별 개수, 패인 흉터 개수와 평균 확률)
        """
        im = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        W, H = im.size
        r = self.yolo.predict(im, conf=conf, imgsz=640, device=self.device, verbose=False)[0]
        names = self.yolo.names
        dets = sorted(zip(r.boxes.conf.tolist(), r.boxes.cls.tolist(), r.boxes.xyxy.tolist()), reverse=True)[:max_lesions]

        crops, metas = [], []
        for c, k, (x1, y1, x2, y2) in dets:
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(W, int(round(x2))), min(H, int(round(y2)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            crops.append(self.tf(im.crop((x1, y1, x2, y2))))
            metas.append((float(c), names[int(k)], [x1, y1, x2, y2]))

        probs = []
        with torch.no_grad():
            for i in range(0, len(crops), 32):
                logits = self.vit(torch.stack(crops[i:i + 32]).to(self.device))
                probs += torch.softmax(logits, dim=1).cpu().tolist()

        lesions = []
        for (c, det_name, box), p in zip(metas, probs):
            by_class = dict(zip(VIT_CLASSES, p))
            top = max(by_class, key=by_class.get)
            lesions.append({
                "box": box,
                "det_conf": round(c, 4),
                "det_class": LABELS.get(det_name, det_name),
                "class": LABELS[top],
                "class_ko": LABELS_KO[top],
                "prob": round(by_class[top], 4),
                "probs": {LABELS[k]: round(v, 4) for k, v in by_class.items()},
                "atrophic_prob": round(sum(by_class[k] for k in ATROPHIC), 4),
            })

        atrophic = [x for x in lesions if x["class"] in ("Boxcar", "Ice pick", "Rolling")]
        counts = {}
        for x in lesions:
            counts[x["class"]] = counts.get(x["class"], 0) + 1
        summary = {
            "n_lesions": len(lesions),
            "counts": counts,
            "n_atrophic": len(atrophic),
            "mean_atrophic_prob": round(sum(x["atrophic_prob"] for x in atrophic) / len(atrophic), 4) if atrophic else 0.0,
        }
        return {"image_size": [W, H], "conf_threshold": conf, "stage2_model": self.stage2, "lesions": lesions, "summary": summary}

    @staticmethod
    def draw(image, result, out_path=None):
        """탐지 상자와 2단계 예측 유형·확률을 그린 영상을 돌려준다."""
        im = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        d = ImageDraw.Draw(im)
        for x in result["lesions"]:
            color = COLORS.get(x["class"], "#000000")
            d.rectangle(x["box"], outline=color, width=2)
            d.text((x["box"][0] + 2, max(0, x["box"][1] - 11)), "%s %.0f%%" % (x["class"], x["prob"] * 100), fill=color)
        if out_path:
            im.save(out_path)
        return im
