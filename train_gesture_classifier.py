"""
Addestra il classificatore del gesto di aiuto ("Signal for Help") con lo stesso
metodo del vecchio progetto Telecamera (train.py): RandomForest a 150 alberi
sulle 630 coordinate grezze delle sequenze.

Dataset: gesture_dataset/dataset_seq.csv. Ogni riga è una sequenza di 15 frame;
per ogni frame i 21 landmark MediaPipe Hands come (x - x_polso, y - y_polso),
registrati con l'immagine specchiata -> 630 feature + "label" ("aiuto"/"normale").

Uso:  venv/bin/python train_gesture_classifier.py
Output: gesture_model.pkl (percorso in config.HELP_GESTURE_MODEL_PATH)
"""
import csv
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import config

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "gesture_dataset", "dataset_seq.csv")
OUT = os.path.join(HERE, getattr(config, "HELP_GESTURE_MODEL_PATH", "gesture_model.pkl"))
SEQ_LEN = getattr(config, "HELP_GESTURE_SEQUENCE_LENGTH", 15)


def main():
    with open(DATASET, newline="") as f:
        rows = list(csv.reader(f))[1:]
    X = np.array([[float(v) for v in r[:-1]] for r in rows], dtype=np.float32)
    y = np.array([r[-1].strip() for r in rows])
    print(f"Sequenze: {len(y)} (aiuto {int((y == 'aiuto').sum())}, normale {int((y == 'normale').sum())})")

    # Solo per il report: 80/20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=150, n_jobs=4, random_state=42)
    clf.fit(X_train, y_train)
    print(classification_report(y_test, clf.predict(X_test), digits=3))

    # Modello finale su tutto il dataset (come train.py)
    clf.fit(X, y)
    with open(OUT, "wb") as f:
        pickle.dump({"model": clf, "sequence_length": SEQ_LEN}, f)
    print(f"Modello salvato in {OUT}")


if __name__ == "__main__":
    main()
