import os
import re
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# CONFIG 
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # use Apple MPS if available

# 5 class
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]  # class order must match training

IMG_SIZE = 224   # ResNet18 expects 224x224 input
SEQ_LEN  = 16    # number of frames sampled per video

# Path
ROOT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "emotion_resnet_lstm_best.pth")
VAL_JSON   = os.path.join(ROOT_DIR, "data", "val.json")
OUT_CSV    = os.path.join(ROOT_DIR, "resnet_lstm_val_results.csv")
OUT_REPORT = os.path.join(ROOT_DIR, "resnet_lstm_classification_report.txt")


# MODEL 
class VideoResNetLSTM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 256)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)          # flatten batch and time for CNN processing
        feats = self.backbone(x).view(b, t, -1)  # extract per-frame features, reshape back to sequence
        lstm_out, _ = self.lstm(feats)            # model temporal relationships across frames
        return self.classifier(lstm_out[:, -1, :])  # use last LSTM output for final prediction


def get_modality(video_name):
    """
    RAVDESS video amodality.
    actor02_01-01-05-...  → speech (01)
    actor02_02-01-05-...  → song   (02)
    """
    # actor prefix
    stem = video_name.split("_", 1)[-1] if "_" in video_name else video_name
    parts = stem.split("-")
    if parts[0] == "01":
        return "speech"
    elif parts[0] == "02":
        return "song"
    return "unknown"


def run_predict_val():
    if not os.path.exists(MODEL_PATH):
        print("Model not found:", MODEL_PATH)
        return

    model = VideoResNetLSTM(num_classes=len(EMOTIONS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Model uploaded: {MODEL_PATH}")
    print(f"Classes ({len(EMOTIONS)}): {EMOTIONS}")
    print(f"Device: {DEVICE}\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(VAL_JSON):
        print("val.json not found:", VAL_JSON)
        return

    with open(VAL_JSON, "r") as f:
        val_items = json.load(f)

    # Video based grouping
    video_groups = defaultdict(list)
    for it in val_items:
        video_groups[it["video"]].append(it)

    print(f"Number of videos to test: {len(video_groups)}\n")

    results    = []
    all_true   = []
    all_pred   = []

    for vid, frames_info in tqdm(video_groups.items()):
        # Frame order
        frames_info = sorted(
            frames_info,
            key=lambda x: int(re.findall(r"\d+", os.path.basename(x["path"]))[-1])
        )

        # Equal SEQ_LEN frame
        indices = np.linspace(0, len(frames_info) - 1, SEQ_LEN, dtype=int)  # evenly spaced frame indices

        frames = []
        for i in indices:
            img_path = os.path.join(ROOT_DIR, frames_info[i]["path"].replace("\\", "/"))
            try:
                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                frames.append(transform(img))
            except Exception:
                frames.append(torch.zeros(3, IMG_SIZE, IMG_SIZE))  # fallback: blank frame if file missing

        x = torch.stack(frames).unsqueeze(0).to(DEVICE)  # (1, 16, 3, 224, 224)

        with torch.no_grad():                          # no gradient needed during inference
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)      # convert raw scores to probabilities
            conf, pred_idx = torch.max(probs, 1)       # get highest probability and its class index

        pred_label = EMOTIONS[pred_idx.item()]
        true_label = frames_info[0]["label"]
        modality   = get_modality(vid)
        is_correct = (pred_label == true_label)

        all_true.append(true_label)
        all_pred.append(pred_label)


        results.append({
            "Video":           vid,
            "Modality":        modality,
            "True_Label":      true_label.upper(),
            "Predicted_Label": pred_label.upper(),
            "Confidence_%":    f"{conf.item()*100:.2f}",
            "Correct":         "YES" if is_correct else "NO",
        })

   
    # ACCURACY
   
    overall_acc = accuracy_score(all_true, all_pred) * 100
    print(f"\n{'='*55}")
    print(f"  RESNET-LSTM VAL ACCURACY: %{overall_acc:.2f}")
    print(f"{'='*55}\n")

    # PER-CLASS ACC
    print("Per-class accuracy:")
    for emo in EMOTIONS:
        idxs = [i for i, t in enumerate(all_true) if t == emo]
        if not idxs:
            print(f"  {emo:10s}: — (no data)")
            continue
        correct = sum(1 for i in idxs if all_pred[i] == emo)
        print(f"  {emo:10s}: %{correct/len(idxs)*100:.1f}  ({correct}/{len(idxs)})")

    # SPEECH vs SONG ACC
    
    df = pd.DataFrame(results)
    print("\nSpeech vs Song accuracy:")
    for mod in ["speech", "song"]:
        sub = df[df["Modality"] == mod]
        if sub.empty:
            continue
        mod_acc = (sub["Correct"] == "YES").sum() / len(sub) * 100
        print(f"  {mod:8s}: %{mod_acc:.1f}  ({len(sub)} video)")

   
    # CLASSIFICATION(Precision / Recall / F1)
   
    report = classification_report(all_true, all_pred, labels=EMOTIONS, target_names=EMOTIONS, digits=3)
    print(f"\nClassification Report:\n{report}")

    
    # CONFUSION MATRIX
   
    cm = confusion_matrix(all_true, all_pred, labels=EMOTIONS)
    print("Confusion Matrix (row=real, col=predict):")
    header = f"{'':12s}" + "".join(f"{e[:7]:>8s}" for e in EMOTIONS)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {EMOTIONS[i]:10s}" + "".join(f"{v:8d}" for v in row))

    
    #file save
    
    df.to_csv(OUT_CSV, index=False)
    print(f"\nCSV saved: {OUT_CSV}")

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(f"RESNET-LSTM VAL ACCURACY: %{overall_acc:.2f}\n\n")
        f.write("Per-class accuracy:\n")
        for emo in EMOTIONS:
            idxs = [i for i, t in enumerate(all_true) if t == emo]
            if idxs:
                correct = sum(1 for i in idxs if all_pred[i] == emo)
                f.write(f"  {emo}: %{correct/len(idxs)*100:.1f} ({correct}/{len(idxs)})\n")
        f.write("\nSpeech vs Song:\n")
        for mod in ["speech", "song"]:
            sub = df[df["Modality"] == mod]
            if not sub.empty:
                mod_acc = (sub["Correct"] == "YES").sum() / len(sub) * 100
                f.write(f"  {mod}: %{mod_acc:.1f} ({len(sub)} video)\n")
        f.write(f"\nClassification Report:\n{report}\n")
        f.write("Confusion Matrix (line=real, line=prediction):\n")
        f.write(header + "\n")
        for i, row in enumerate(cm):
            f.write(f"  {EMOTIONS[i]:10s}" + "".join(f"{v:8d}" for v in row) + "\n")

    print(f"file saved: {OUT_REPORT}")


if __name__ == "__main__":
    run_predict_val()
