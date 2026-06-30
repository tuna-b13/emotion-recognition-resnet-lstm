"""
Manuel Test — Tek/Çoklu Video Inference
----------------------------------------
Kullanım:
    python data/manuel_test/predict_manuel.py

videos/ klasörüne herhangi bir kaynak videosunu (.mp4 / .mov / .avi) at,
script modeli yükleyip her video için duygu tahmini yapar.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ---------- PATHS ----------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "emotion_resnet_lstm_best.pth")
VIDEOS_DIR = os.path.join(THIS_DIR, "videos")

# ---------- CONFIG ----------
DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]
IMG_SIZE = 224
SEQ_LEN  = 16

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ---------- MODEL ----------
class VideoResNetLSTM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256,
                            num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x).view(b, t, -1)
        lstm_out, _ = self.lstm(feats)
        return self.classifier(lstm_out[:, -1, :])


# ---------- FACE EXTRACTION ----------
def extract_frames(video_path: str, n_frames: int = SEQ_LEN):
    """Videodan eşit aralıklı n_frames kare çıkarır, yüz varsa kırpar."""
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)
    indices = set(np.linspace(0, total - 1, n_frames, dtype=int).tolist())

    raw_frames = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in indices:
            raw_frames[idx] = frame
        idx += 1
    cap.release()

    frames = []
    for i in sorted(raw_frames.keys()):
        frame = raw_frames[i]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(48, 48))
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            crop = frame[y:y+h, x:x+w]
        else:
            crop = frame  # yüz bulunamazsa tüm kareyi kullan

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        frames.append(img)

    # eksik frame varsa son kareyi tekrarla
    while len(frames) < n_frames:
        frames.append(frames[-1] if frames else np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    return frames[:n_frames]


# ---------- INFERENCE ----------
def predict_video(model, video_path: str, transform):
    frames_np = extract_frames(video_path)
    tensors = [transform(Image.fromarray(f)) for f in frames_np]
    x = torch.stack(tensors).unsqueeze(0).to(DEVICE)  # (1, 16, 3, 224, 224)

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, 0)

    label = EMOTIONS[idx.item()]
    all_probs = {e: round(probs[i].item() * 100, 1) for i, e in enumerate(EMOTIONS)}
    return label, round(conf.item() * 100, 1), all_probs


# ---------- GROUND TRUTH PARSING ----------
# Dosya adı formatı: 1003_IEO_ANG_HI.mp4  →  3. parça = duygu kodu
LABEL_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "FEA": "fearful",
    "DIS": "disgust",
    "SUR": "surprise",
}

def parse_ground_truth(filename: str):
    """Dosya adından gerçek etiketi döndürür; bulunamazsa None."""
    parts = os.path.splitext(filename)[0].split("_")
    for p in parts:
        if p.upper() in LABEL_MAP:
            return LABEL_MAP[p.upper()]
    return None


# ---------- VISUALIZATION ----------
def save_results_figure(results, out_dir):
    """Sonuçları görsel olarak kaydeder: tablo + confusion matrix + per-class accuracy."""
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Tablo görseli ----
    fig, ax = plt.subplots(figsize=(14, max(4, len(results) * 0.35 + 1)))
    ax.axis("off")
    col_labels = ["Video", "Gercek", "Tahmin", "Guven", "Dogru?"]
    table_data = []
    for r in results:
        correct = "✓" if r["correct"] is True else ("✗" if r["correct"] is False else "?")
        table_data.append([
            r["video"], r["truth"] or "?", r["pred"].upper(),
            f"{r['conf']:.1f}%", correct
        ])
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.3)
    # Satır renklendirme
    for i, r in enumerate(results):
        color = "#d4edda" if r["correct"] is True else ("#f8d7da" if r["correct"] is False else "#fff3cd")
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(color)
    plt.title("Manuel Test Sonuclari", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    tbl_path = os.path.join(out_dir, "results_table.png")
    plt.savefig(tbl_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Gorsel] Tablo kaydedildi: {tbl_path}")

    # Ground truth mevcut olan sonuçları filtrele
    labeled = [r for r in results if r["truth"] is not None]
    if not labeled:
        print("[Uyari] Hicbir videoda ground truth bulunamadi; confusion matrix atlanıyor.")
        return

    y_true = [r["truth"] for r in labeled]
    y_pred = [r["pred"]  for r in labeled]
    classes = sorted(set(y_true) | set(y_pred))

    # ---- 2. Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[Gorsel] Confusion matrix kaydedildi: {cm_path}")

    # ---- 3. Per-class accuracy bar ----
    per_class = {}
    for cls in classes:
        total   = sum(1 for r in labeled if r["truth"] == cls)
        correct = sum(1 for r in labeled if r["truth"] == cls and r["pred"] == cls)
        per_class[cls] = (correct / total * 100) if total else 0

    overall_acc = sum(1 for r in labeled if r["correct"]) / len(labeled) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(per_class.keys(), per_class.values(),
                  color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
    ax.axhline(overall_acc, color="red", linestyle="--", linewidth=1.5,
               label=f"Overall Acc: {overall_acc:.1f}%")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.legend()
    for bar, val in zip(bars, per_class.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}%", ha="center", fontsize=10)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, "per_class_accuracy.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[Gorsel] Per-class accuracy kaydedildi: {bar_path}")

    # ---- Özet metin ----
    print(f"\n{'='*50}")
    print(f"OVERALL ACCURACY : {overall_acc:.1f}%  ({sum(r['correct'] for r in labeled)}/{len(labeled)})")
    print(f"{'='*50}")
    print("\nPer-class accuracy:")
    for cls, acc in per_class.items():
        total = sum(1 for r in labeled if r["truth"] == cls)
        correct = sum(1 for r in labeled if r["truth"] == cls and r["pred"] == cls)
        print(f"  {cls:<10} {acc:5.1f}%  ({correct}/{total})")
    print(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=classes)}")


# ---------- MAIN ----------
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[HATA] Model bulunamadi: {MODEL_PATH}")
        sys.exit(1)

    model = VideoResNetLSTM(num_classes=len(EMOTIONS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model yuklendi: {MODEL_PATH}")
    print(f"Device: {DEVICE}\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # videos/ klasöründeki tüm videoları bul
    exts = (".mp4", ".mov", ".avi", ".mkv")
    videos = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(exts)]

    if not videos:
        print(f"videos/ klasoru bos. Videolarini su klasore at:\n  {VIDEOS_DIR}")
        return

    print(f"{len(videos)} video bulundu.\n")
    print(f"{'Video':<40} {'Gercek':<10} {'Tahmin':<10} {'Guven':>6}  Tum olasiliklar")
    print("-" * 100)

    results = []
    for vname in sorted(videos):
        vpath = os.path.join(VIDEOS_DIR, vname)
        truth = parse_ground_truth(vname)
        try:
            label, conf, all_probs = predict_video(model, vpath, transform)
            correct = (label == truth) if truth else None
            probs_str = "  ".join(f"{e}:{v:.0f}%" for e, v in all_probs.items())
            marker = " ✓" if correct is True else (" ✗" if correct is False else "")
            print(f"{vname:<40} {(truth or '?').upper():<10} {label.upper():<10} {conf:>5.1f}%{marker}  {probs_str}")
            results.append({"video": vname, "truth": truth, "pred": label,
                            "conf": conf, "correct": correct})
        except Exception as e:
            print(f"{vname:<40} [HATA] {e}")

    # Görselleri kaydet
    out_dir = os.path.join(THIS_DIR, "results_plots")
    save_results_figure(results, out_dir)


if __name__ == "__main__":
    main()
