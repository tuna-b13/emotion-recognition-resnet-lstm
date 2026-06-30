"""
Real-Time Emotion Recognition
PALANTIR / MATRIX  HUD  ─  ResNet-LSTM
---------------------------------------
    python ce903-emotion/realtime_demo.py
'q' to quit.
"""
import os, sys, time, math, random
import torch, torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2, numpy as np
from collections import deque

#  PATHS 
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, "models", "emotion_resnet_lstm_best.pth")

# CONFIG 
DEVICE   = torch.device("mps" if torch.backends.mps.is_available() else
                        ("cuda" if torch.cuda.is_available() else "cpu"))
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful"]
IMG_SIZE = 224
SEQ_LEN  = 16
PANEL_W  = 310   # right data panel width

#  PALETTE  
C_GREEN  = ( 0, 255,  65)   # Matrix green
C_DIM    = ( 0,  65,  18)   # dim green
C_DARK   = ( 3,  10,   3)   # near-black panel bg
C_BRIGHT = (200, 255, 210)  # bright highlight
C_LINE   = ( 0,  55,  14)   # separator line
C_AMBER  = ( 0, 180, 255)   # warning orange

EMO_COLOR = {
    "neutral": ( 0, 200,  55),
    "happy":   ( 0, 255,  65),
    "sad":     (160, 110,   0),
    "angry":   ( 0,  70, 240),
    "fearful": ( 30,  30, 210),
}
EMO_TAG = {
    "neutral": "NEUTRAL    AFFECT",
    "happy":   "POSITIVE   AFFECT",
    "sad":     "NEGATIVE   AFFECT",
    "angry":   "[!] HOSTILE AFFECT",
    "fearful": "[!] THREAT  RESP.",
}
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# MODEL
class VideoResNetLSTM(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 256)
        self.lstm = nn.LSTM(256, 256, num_layers=2, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(256, num_classes))

    def forward(self, x):
        b, t, c, h, w = x.shape
        feats = self.backbone(x.view(b*t, c, h, w)).view(b, t, -1)
        out, _ = self.lstm(feats)
        return self.classifier(out[:, -1, :])


#  MATRIX RAIN 
class MatrixRain:
    _CH = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&*-+=<>|!?")

    def __init__(self, width, height, cw=9, ch=13):
        self.w, self.h, self.cw, self.ch = width, height, cw, ch
        n = width // cw
        self.drops  = [random.randint(0, height // ch) for _ in range(n)]
        self.speeds = [random.choice([0.4, 0.6, 1.0, 1.0, 1.5]) for _ in range(n)]
        self._buf   = np.zeros((height, width, 3), np.uint8)

    def render(self) -> np.ndarray:
        self._buf = (self._buf * 0.80).astype(np.uint8)
        n = self.w // self.cw
        for i in range(n):
            x = i * self.cw
            y = int(self.drops[i]) * self.ch
            if 0 <= y < self.h:
                cv2.putText(self._buf, random.choice(self._CH),
                            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                            C_BRIGHT, 1, cv2.LINE_AA)
            self.drops[i] += self.speeds[i]
            if self.drops[i] * self.ch > self.h + 20:
                if random.random() < 0.05:
                    self.drops[i] = 0
        return self._buf


# HUD PRIMITIVES 
def alpha_rect(dst, pt1, pt2, color, a=0.82):
    ov = dst.copy()
    cv2.rectangle(ov, pt1, pt2, color, -1)
    cv2.addWeighted(ov, a, dst, 1-a, 0, dst)

def mono(f, txt, x, y, sc=0.37, col=C_GREEN, bold=False):
    cv2.putText(f, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2 if bold else 1, cv2.LINE_AA)

def seg_bar(f, x, y, w, h, frac, col, n=20, gap=2):
    sw = max(1, (w - gap*(n-1)) // n)
    filled = int(frac * n)
    for i in range(n):
        x0 = x + i*(sw+gap)
        cv2.rectangle(f, (x0, y), (x0+sw, y+h), col if i < filled else C_DIM, -1)

def hline(f, x1, x2, y, col=C_LINE):
    cv2.line(f, (x1, y), (x2, y), col, 1)

def scanlines(f, step=3, a=0.07):
    ov = f.copy()
    for y in range(0, f.shape[0], step):
        cv2.line(ov, (0, y), (f.shape[1], y), (0, 0, 0), 1)
    cv2.addWeighted(ov, a, f, 1-a, 0, f)

def frame_corners(f, col, arm=30, t=2):
    """HUD corner brackets on the full video viewport."""
    ih, iw = f.shape[:2]
    vw = iw - PANEL_W
    for (x, y, dx, dy) in [(0,0,1,1),(vw,0,-1,1),(0,ih,1,-1),(vw,ih,-1,-1)]:
        cv2.line(f, (x, y), (x+dx*arm, y), col, t)
        cv2.line(f, (x, y), (x, y+dy*arm), col, t)

def targeting_reticle(f, x, y, w, h, col, phase):
    """Military targeting: corner brackets + animated scan + conf arc."""
    arm, t = 26, 2
    # Corner L-brackets
    for (cx,cy,dx,dy) in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(f, (cx, cy), (cx+dx*arm, cy), col, t)
        cv2.line(f, (cx, cy), (cx, cy+dy*arm), col, t)
    # Animated horizontal scan
    sy = y + int(((math.sin(phase) + 1) / 2) * h)
    ov = f.copy()
    cv2.line(ov, (x, sy), (x+w, sy), col, 1)
    cv2.addWeighted(ov, 0.55, f, 0.45, 0, f)
    # Center crosshair
    cx, cy = x + w//2, y + h//2
    cv2.line(f, (cx-6, cy), (cx+6, cy), col, 1)
    cv2.line(f, (cx, cy-6), (cx, cy+6), col, 1)
    cv2.circle(f, (cx, cy), 3, col, -1)

def conf_arc(f, x, y, w, h, conf, col):
    """Partial ellipse arc around face showing confidence."""
    cx, cy = x+w//2, y+h//2
    r = max(w, h)//2 + 22
    cv2.ellipse(f, (cx, cy), (r, r), -90, 0, int(conf*359), col, 2)
    # Small tick marks at 0% and 100% position
    cv2.line(f, (cx, cy-r-4), (cx, cy-r+4), col, 1)

def face_label(f, x, y, w, emotion, conf, col):
    """Label bubble below face box."""
    txt = f"{emotion.upper()}  {conf*100:.0f}%"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
    bx, by = x, y - th - 14
    # Small filled bar, no rounded—sharp angular style
    alpha_rect(f, (bx-2, by-4), (bx+tw+10, by+th+4), (0, 15, 0), 0.85)
    cv2.rectangle(f, (bx-2, by-4), (bx+tw+10, by+th+4), col, 1)
    # Connector tick
    cv2.line(f, (bx+tw//2, y), (bx+tw//2, by+th+4), col, 1)
    mono(f, txt, bx+4, by+th-1, 0.50, col, bold=True)


# RIGHT PANEL
def draw_panel(f, probs, emotion, conf, buf_n, fps, ih, iw, rain: MatrixRain):
    px   = iw - PANEL_W
    col  = EMO_COLOR.get(emotion, C_GREEN)
    t    = time.time()
    blink = int(t * 2) % 2 == 0

    # Panel BG 
    alpha_rect(f, (px, 0), (iw, ih), C_DARK, 0.45)

    # Left accent border (emotion color)
    cv2.rectangle(f, (px, 0), (px+2, ih), col, -1)

    #  HEADER 
    alpha_rect(f, (px+2, 0), (iw, 42), (0, 20, 0), 0.60)
    mono(f, "BIOMETRIC  ANALYSIS  SYS", px+10, 15, 0.34, C_GREEN)
    mono(f, f"v2.1  |  {str(DEVICE).upper()}  |  {fps:4.0f} FPS", px+10, 30, 0.28, C_DIM)
    # Blink indicator
    cv2.circle(f, (iw-16, 18), 5, C_GREEN if blink else C_DIM, -1)
    hline(f, px+2, iw, 42, C_LINE)

    # ── SUBJECT STATUS
    y = 58
    mono(f, "SUBJECT", px+10, y, 0.28, C_DIM)
    if emotion != "waiting...":
        mono(f, "DETECTED", px+10, y+16, 0.42, col, bold=True)
    else:
        dots = "." * (int(t*3) % 4)
        mono(f, f"SCANNING{dots:<4}", px+10, y+16, 0.42, C_AMBER)
    hline(f, px+8, iw-8, y+26, C_LINE)

    # PRIMARY CLASSIFICATION 
    mono(f, "PRIMARY  CLASSIFICATION", px+10, y, 0.28, C_DIM)
    if emotion != "waiting...":
        tag = EMO_TAG.get(emotion, emotion.upper())
        mono(f, tag, px+10, y+18, 0.44, col, bold=True)
        mono(f, "CONFIDENCE", px+10, y+36, 0.28, C_DIM)
        seg_bar(f, px+10, y+40, PANEL_W-20, 9, conf, col, n=25, gap=2)
        mono(f, f"{conf*100:05.1f}%", iw-58, y+49, 0.36, col)
    else:
        anim = "." * (int(t*2.5) % 5)
        mono(f, f"ANALYZING{anim:<5}", px+10, y+18, 0.42, C_DIM)
    hline(f, px+8, iw-8, y+56, C_LINE)

    #  PROBABILITY MATRIX 
    y = 165
    mono(f, "AFFECT  PROBABILITY  MATRIX", px+10, y, 0.28, C_DIM)
    y += 12
    for emo in EMOTIONS:
        p      = probs.get(emo, 0.0)
        ec     = EMO_COLOR.get(emo, C_GREEN)
        is_top = (emo == emotion)
        if is_top:
            alpha_rect(f, (px+4, y), (iw-4, y+16), ec, 0.10)
        mono(f, f"{emo.upper():<8}", px+10, y+12,
             0.31, col if is_top else C_DIM, bold=is_top)
        bx = px + 84;  bw = PANEL_W - 104
        alpha_rect(f, (bx, y+3), (bx+bw, y+11), (0, 20, 5), 0.50)
        fw = int(bw * p)
        if fw > 0:
            cv2.rectangle(f, (bx, y+3), (bx+fw, y+11), ec, -1)
        mono(f, f"{p*100:4.0f}%", bx+bw+4, y+12, 0.31,
             col if is_top else C_DIM)
        y += 20
    hline(f, px+8, iw-8, y+4, C_LINE)

    # SEQUENCE BUFFER 
    mono(f, "SEQUENCE  BUFFER", px+10, y, 0.28, C_DIM)
    y += 12
    buf_col = C_GREEN if buf_n == SEQ_LEN else C_AMBER
    seg_bar(f, px+10, y, PANEL_W-20, 8, buf_n/SEQ_LEN, buf_col, n=SEQ_LEN, gap=2)
    mono(f, f"{buf_n:02d}/{SEQ_LEN:02d}  FRAMES  LOADED", px+10, y+20, 0.28, C_DIM)
    hline(f, px+8, iw-8, y+26, C_LINE)

    # FOOTER 
    alpha_rect(f, (px+2, ih-38), (iw, ih), (0, 14, 0), 0.60)
    ts = time.strftime("%Y-%m-%d  %H:%M:%S")
    mono(f, ts, px+10, ih-22, 0.29, C_DIM)
    mono(f, "ResNet18 + LSTM  |  CE903  EMOTION  AI", px+10, ih-8, 0.26, C_DIM)


#  MAIN 
    if not os.path.exists(MODEL_PATH):
        print(f"[HATA] Model bulunamadi: {MODEL_PATH}"); sys.exit(1)

    model = VideoResNetLSTM(num_classes=len(EMOTIONS)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    print(f"Model  : {MODEL_PATH}")
    print(f"Device : {DEVICE}")
    print("'q' ile cikisin.\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    face_cascade = cv2.CascadeClassifier(CASCADE)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera acilamadi."); sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    ih_init = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    iw_init = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    rain = MatrixRain(PANEL_W, ih_init)

    frame_buf  = deque(maxlen=SEQ_LEN)
    prob_hist  = deque(maxlen=3)
    last_emo   = "waiting..."
    last_conf  = 0.0
    last_probs = {e: 0.0 for e in EMOTIONS}
    last_face  = None

    frame_idx = 0
    t_prev    = time.time()
    fps       = 0.0
    scan_phase = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        ih, iw = frame.shape[:2]

        # FPS
        now    = time.time()
        fps    = 0.92*fps + 0.08*(1.0/max(now-t_prev, 1e-5))
        t_prev = now
        scan_phase += 0.12

        # Face detection
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(64,64))

        face_tensor = None
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
            x, y = max(0, x), max(0, y)
            w = min(w, iw-PANEL_W-x);  h = min(h, ih-y)
            last_face = (x, y, w, h)
            if w > 0 and h > 0:
                crop = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                face_tensor = transform(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)))

        if face_tensor is None:
            face_tensor = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        frame_buf.append(face_tensor)

        # Inference every 2 frames
        if len(frame_buf) == SEQ_LEN and frame_idx % 2 == 0:
            inp = torch.stack(list(frame_buf)).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(model(inp), 1)[0].cpu().numpy()
            prob_hist.append(probs)
            sm = np.mean(prob_hist, axis=0)
            idx = int(np.argmax(sm))
            last_emo   = EMOTIONS[idx]
            last_conf  = float(sm[idx])
            last_probs = {e: float(sm[i]) for i, e in enumerate(EMOTIONS)}

        #  RENDER 
        col = EMO_COLOR.get(last_emo, C_GREEN)

        # Right panel
        draw_panel(frame, last_probs, last_emo, last_conf,
                   len(frame_buf), fps, ih, iw, rain)

        # Video viewport corner brackets
        frame_corners(frame, C_DIM, arm=28)

        # Face targeting
        if last_face is not None and len(faces) > 0:
            x, y, w, h = last_face
            targeting_reticle(frame, x, y, w, h, col, scan_phase)
            conf_arc(frame, x, y, w, h, last_conf, col)
            face_label(frame, x, y, w, last_emo, last_conf, col)

        # No-face indicator
        if len(faces) == 0 and last_emo != "waiting...":
            mono(frame, "NO SUBJECT DETECTED", 20, ih//2,
                 0.55, C_DIM, bold=False)

        cv2.imshow("BIOMETRIC ANALYSIS  |  ResNet+LSTM  |  q: quit", frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
