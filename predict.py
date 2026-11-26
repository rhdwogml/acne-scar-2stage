"""명령줄에서 한 장을 추론한다.

    python predict.py face.jpg                  # 결과를 JSON 으로 출력
    python predict.py face.jpg --out out.jpg    # 상자를 그린 영상도 저장
    python predict.py face.jpg --conf 0.15      # 탐지 임계값을 낮춰 더 많이 잡기
"""
import argparse
import json
import sys

from scar_pipeline import ScarPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--conf", type=float, default=0.25, help="1단계 탐지 신뢰도 임계값 (기본 0.25)")
    ap.add_argument("--out", default=None, help="상자를 그린 영상을 저장할 경로")
    ap.add_argument("--device", default=None, help="cpu 또는 cuda (기본: 자동)")
    ap.add_argument("--stage2", default="auto", choices=["auto", "resnet50", "vit_cbam"], help="2단계 분류 모델 (기본: resnet50 가중치가 있으면 resnet50)")
    a = ap.parse_args()

    pipe = ScarPipeline(stage2=a.stage2, device=a.device)
    result = pipe.predict(a.image, conf=a.conf)
    if a.out:
        pipe.draw(a.image, result, a.out)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
