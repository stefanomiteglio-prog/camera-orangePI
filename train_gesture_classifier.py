import glob, os, random
import numpy as np
import cv2
import mediapipe as mp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

BASE = "/home/orangepi/treeyes_datasets/signal_for_help/3 Classification"
POS_DIR = os.path.join(BASE, "2. Signal for Help")
NEG_DIR = os.path.join(BASE, "1. No Signal")
MAX_PER_CLASS = 400
RESIZE_TO = 320

random.seed(42)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)


def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = RESIZE_TO / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    diag = max(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    ox, oy = lm[0].x, lm[0].y
    feats = []
    for p in lm:
        feats.append((p.x - ox) / diag)
        feats.append((p.y - oy) / diag)
    return np.array(feats, dtype=np.float32)


X, y = [], []
for label, folder in [(1, POS_DIR), (0, NEG_DIR)]:
    paths = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png"))
    random.shuffle(paths)
    paths = paths[:MAX_PER_CLASS]
    print(f"Classe {label}: {len(paths)} immagini campionate da {folder}", flush=True)
    ok_count = 0
    for i, p in enumerate(paths):
        f = extract_features(p)
        if f is not None:
            X.append(f)
            y.append(label)
            ok_count += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)} processate...", flush=True)
    print(f"  -> mano rilevata in {ok_count}/{len(paths)} immagini", flush=True)

X = np.array(X)
y = np.array(y)
print(f"\nDataset finale: {X.shape[0]} esempi, {X.shape[1]} feature")
print(f"Positivi: {(y==1).sum()}, Negativi: {(y==0).sum()}")

if len(set(y)) < 2:
    print("ERRORE: servono entrambe le classi per allenare il classificatore.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n--- Report su test set ---")
print(classification_report(y_test, y_pred, target_names=["no_signal", "signal_for_help"]))

with open("/home/orangepi/treeyes_vision/gesture_classifier.pkl", "wb") as f:
    pickle.dump(clf, f)
print("\nModello salvato in gesture_classifier.pkl")
