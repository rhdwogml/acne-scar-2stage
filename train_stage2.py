"""2단계 분류 모델을 학습하고 가중치를 weights/ 에 저장한다. (GPU 권장)

    python train_stage2.py --data C:\\path\\to\\scar_final_cls --model resnet50 --seed 2

--data 폴더 안에 Train / Val / Test 가 있고, 그 아래에 클래스별 폴더가 있어야 한다.
학습 설정은 논문 수정 실험과 같다: 224 x 224, 배치 16, Adam 1e-4, 최대 50 에폭,
클래스 빈도 역수 가중 교차 엔트로피, 검증 macro-F1 기준 조기 종료(patience 10).

결과: weights/<model>_scar.pt (가중치), weights/<model>_scar.json (시험 집합 성능)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from scar_pipeline.model import IMG_SIZE, build_stage2
from scar_pipeline.pipeline import VIT_CLASSES

HERE = Path(__file__).resolve().parent
BATCH, EPOCHS, LR, PATIENCE = 16, 50, 1e-4, 10


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="scar_final_cls 폴더 (Train/Val/Test 포함)")
    ap.add_argument("--model", default="resnet50", choices=["resnet50", "vit_cbam"])
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    data = Path(a.data)
    if not (data / "Train").exists():
        sys.exit("[오류] 데이터가 없습니다: %s\\Train" % data)
    out_w = HERE / "weights" / ("%s_scar.pt" % a.model)
    out_j = HERE / "weights" / ("%s_scar.json" % a.model)

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("model:", a.model, "| seed:", a.seed, "| device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "(GPU 없음 - 매우 느림)")

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(),
                                   transforms.RandomRotation(20), transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), norm])
    train_set = datasets.ImageFolder(str(data / "Train"), train_tf)
    val_set = datasets.ImageFolder(str(data / "Val"), eval_tf)
    test_set = datasets.ImageFolder(str(data / "Test"), eval_tf)
    classes = train_set.classes
    if classes != VIT_CLASSES:
        sys.exit("[오류] 클래스 폴더가 추론 코드의 순서와 다릅니다: %s" % classes)
    nc = len(classes)
    counts = np.bincount(train_set.targets, minlength=nc)
    print("train=%d val=%d test=%d | %s" % (len(train_set), len(val_set), len(test_set), dict(zip(classes, counts.tolist()))))

    g = torch.Generator()
    g.manual_seed(a.seed)
    pin = device == "cuda"
    # 마지막에 1장만 남는 배치는 BatchNorm 오류를 내므로 버린다
    train_dl = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=a.workers, pin_memory=pin,
                          worker_init_fn=seed_worker, generator=g, drop_last=True)
    val_dl = DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=a.workers, pin_memory=pin)
    test_dl = DataLoader(test_set, batch_size=BATCH, shuffle=False, num_workers=a.workers, pin_memory=pin)

    model = build_stage2(a.model, num_classes=nc, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    cw = torch.tensor(counts).float()
    class_weights = (cw.sum() / (nc * cw.clamp(min=1))).to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    @torch.no_grad()
    def evaluate(dl):
        model.eval()
        preds, gts = [], []
        for xb, yb in dl:
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                o = model(xb.to(device))
            preds += o.argmax(1).cpu().tolist()
            gts += yb.tolist()
        return gts, preds

    best_f1, wait, best_ep = -1.0, 0, 0
    tmp = out_w.with_suffix(".tmp")
    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * yb.size(0)
            seen += yb.size(0)
        gts, preds = evaluate(val_dl)
        val_f1 = f1_score(gts, preds, average="macro", zero_division=0)
        improved = val_f1 > best_f1
        print("[epoch %d/%d] train_loss=%.4f val_acc=%.4f val_macroF1=%.4f %s (%.0fs)" % (
            ep, EPOCHS, run_loss / max(1, seen), accuracy_score(gts, preds), val_f1,
            "*best*" if improved else "(no-improve %d/%d)" % (wait + 1, PATIENCE), time.time() - t0), flush=True)
        if improved:
            best_f1, best_ep, wait = val_f1, ep, 0
            torch.save(model.state_dict(), tmp)
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping at epoch %d (best val macroF1=%.4f @ epoch %d)" % (ep, best_f1, best_ep))
                break

    model.load_state_dict(torch.load(tmp, map_location=device))
    gts, preds = evaluate(test_dl)
    labels = list(range(nc))
    rep = classification_report(gts, preds, labels=labels, target_names=classes, zero_division=0, output_dict=True)
    print("\nBEST val macroF1=%.4f @ epoch %d" % (best_f1, best_ep))
    print("TEST acc=%.4f  macroF1=%.4f" % (accuracy_score(gts, preds), f1_score(gts, preds, average="macro", zero_division=0)))
    print(classification_report(gts, preds, labels=labels, target_names=classes, zero_division=0))

    if out_w.exists():
        out_w.unlink()
    tmp.rename(out_w)
    out_j.write_text(json.dumps(dict(
        model=a.model, seed=a.seed, torch=torch.__version__, best_epoch=best_ep, best_val_macro_f1=best_f1,
        n_train=len(train_set), n_val=len(val_set), n_test=len(test_set),
        test_accuracy=accuracy_score(gts, preds), test_macro_f1=f1_score(gts, preds, average="macro", zero_division=0),
        per_class={c: rep[c] for c in classes}, classes=classes,
        confusion_matrix=confusion_matrix(gts, preds, labels=labels).tolist(),
        minutes=(time.time() - t_start) / 60), ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_w, "(%.1f분)" % ((time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
