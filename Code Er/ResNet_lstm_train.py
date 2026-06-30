import os
import re
import json
from collections import defaultdict, Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models



# CONFIGURATIONS

# 5 class fearfull added
# Song and speecha dded
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]  # 5 target emotion classes
LABEL_MAP = {e: i for i, e in enumerate(EMOTIONS)}         

SEQ_LEN = 16     # number of frames
IMG_SIZE = 224   # input image size 

BATCH_SIZE = 8   
EPOCHS = 30      # total training epochs
LR = 5e-5        # learning rate (kept small for fine-tuning pretrained weights)
WEIGHT_DECAY = 1e-4  # L2 regularisation to reduce overfitting

NUM_WORKERS = 0        #  for stabilty
PIN_MEMORY = False     # macOS/MPS close

BEST_PATH = "models/emotion_resnet_lstm_best.pth"  # path to save the best model checkpoint



# DEVICE
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



# MODEL
class VideoResNetLSTM(nn.Module):
    def __init__(self, num_classes=5, feat_dim=256, lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # load pretrained ResNet18
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feat_dim)  # replace final layer to output 256-dim features

        self.lstm = nn.LSTM(
            input_size=feat_dim,    # matches ResNet output dimension
            hidden_size=lstm_hidden,
            num_layers=lstm_layers, # 2-layer LSTM to capture temporal patterns
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),                       # dropout to prevent overfitting
            nn.Linear(lstm_hidden, num_classes)    # final layer: 256 → 5 emotion classes
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x)        # (B*T, feat_dim)
        feats = feats.view(b, t, -1)   # (B, T, feat_dim)
        lstm_out, _ = self.lstm(feats) # (B, T, hidden)
        last = lstm_out[:, -1, :]      # (B, hidden)
        return self.classifier(last)   # (B, num_classes)



# DATASET
class JsonVideoDataset(Dataset):
    def __init__(self, json_items, transform, seq_len=16, project_root=None):
        self.transform = transform
        self.seq_len = seq_len
        self.project_root = project_root or os.getcwd()

        # video based frame gropus
        self.video_groups = defaultdict(list)
        for it in json_items:
            lab = it.get("label")
            vid = it.get("video")
            if lab in LABEL_MAP and vid:
                self.video_groups[vid].append(it)

        self.video_ids = list(self.video_groups.keys())

    def __len__(self):
        return len(self.video_ids)

    def _abs_path(self, p):
        # JSON' normalize 
        p = p.replace("\\", "/")
        return os.path.join(self.project_root, p)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        frames_info = self.video_groups[vid]

        def frame_num(item):
            bn = os.path.basename(item["path"].replace("\\", "/"))
            nums = re.findall(r"\d+", bn)
            return int(nums[-1]) if nums else 0

        frames_info = sorted(frames_info, key=frame_num)   # ensure frames are in temporal order
        label = LABEL_MAP[frames_info[0]["label"]]          # all frames share the same label

        # select equal seq len
        indices = np.linspace(0, len(frames_info) - 1, self.seq_len, dtype=int)  # evenly spaced frame indices
        frames = []
        for i in indices:
            fp = self._abs_path(frames_info[i]["path"])
            img = Image.open(fp).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))   # resize to 224x224 for ResNet input
            frames.append(self.transform(img))

        x = torch.stack(frames)  # (T, C, H, W)
        return x, label



# TRAIN / EVAL

def compute_class_weights(train_items):
    # video-level label
    first = {}
    for it in train_items:
        vid = it.get("video")
        lab = it.get("label")
        if vid and lab in LABEL_MAP:
            first.setdefault(vid, lab)
    c = Counter(first.values())
    total = sum(c.values())
    w = [total / max(1, c.get(e, 1)) for e in EMOTIONS]  # inverse frequency: rarer classes get higher weight
    return torch.tensor(w, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    # per-class 
    class_correct = Counter()
    class_total   = Counter()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            loss_sum += loss.item() * y.size(0)  # accumulate weighted sum for average loss
            preds = logits.argmax(1)              # pick the class with highest score
            correct += (preds == y).sum().item()
            total += y.size(0)
            for true, pred in zip(y.cpu().tolist(), preds.cpu().tolist()):
                class_total[true] += 1
                if true == pred:
                    class_correct[true] += 1

    per_class_acc = {
        EMOTIONS[i]: round(class_correct[i] / max(1, class_total[i]), 3)
        for i in range(len(EMOTIONS))
    }
    return (loss_sum / max(1, total)), (correct / max(1, total)), per_class_acc


def main():
    device = get_device()
    project_root = os.path.dirname(os.path.abspath(__file__))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_json = os.path.join(project_root, "data", "train.json")
    val_json   = os.path.join(project_root, "data", "val.json")

    with open(train_json, "r", encoding="utf-8") as f:
        train_items = json.load(f)
    with open(val_json, "r", encoding="utf-8") as f:
        val_items = json.load(f)

    train_ds = JsonVideoDataset(train_items, transform, seq_len=SEQ_LEN, project_root=project_root)
    val_ds   = JsonVideoDataset(val_items,   transform, seq_len=SEQ_LEN, project_root=project_root)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = VideoResNetLSTM(num_classes=len(EMOTIONS)).to(device)

    class_w = compute_class_weights(train_items).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)  # weighted loss to handle imbalance

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # AdamW

    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_path  = os.path.join(project_root, BEST_PATH)

    print(f"TRAIN START | device={device}")
    print(f"Classes ({len(EMOTIONS)}): {EMOTIONS}")
    print(f"Videos: train={len(train_ds)} val={len(val_ds)} | seq_len={SEQ_LEN} batch={BATCH_SIZE}")
    print(f"Class weights: {class_w.detach().cpu().numpy().round(2).tolist()}")

    best_acc = -1.0
    history = []  #save every epoch
    history_path = os.path.join(project_root, "models", "train_history.json")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)  # clear gradients 
            logits = model(x)                       # forward pass
            loss = criterion(logits, y)             # compute loss
            loss.backward()                         # backpropagation
            optimizer.step()                        
            running += loss.item() * y.size(0)      
            loop.set_postfix(train_loss=(running / max(1, (loop.n + 1) * BATCH_SIZE)))

        train_loss = running / max(1, len(train_ds))
        val_loss, val_acc, per_class_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")
        print(f"  Per-class acc: " + " | ".join(f"{e}={v:.3f}" for e, v in per_class_acc.items()))

        # history save
        history.append({
            "epoch":         epoch,
            "train_loss":    round(train_loss, 4),
            "val_loss":      round(val_loss, 4),
            "val_acc":       round(val_acc, 4),
            "per_class_acc": per_class_acc,
        })
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)  # overwrite best checkpoint only when val_acc improves
            print(f"  Best model saved → {BEST_PATH} (val_acc={best_acc:.3f})")

        torch.save(model.state_dict(), os.path.join(models_dir, f"resnet_lstm_epoch_{epoch}.pth"))  # also save per-epoch checkpoint

    print(f"train completed. History → {history_path}")


if __name__ == "__main__":
    main()
