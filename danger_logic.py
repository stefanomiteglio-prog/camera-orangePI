"""
Logica di conferma dei pericoli (armi), indipendente da NPU e telecamera.

Sostituisce la vecchia regola "visibile per N secondi di fila": il detector
sfarfalla (lama di taglio, mano che ruota, sfocatura) e bastavano due frame
persi per azzerare il conteggio. Qui un pericolo è confermato se, nella finestra
mobile degli ultimi DANGER_WINDOW_SECONDS, è stato visto in una frazione
sufficiente di frame e per una durata minima.

Solo Python standard (gira anche su PC senza numpy per i test).
"""
import math
from collections import deque


class DangerConfirmer:
    """
    Una finestra di osservazioni (istante, score o 0) per ogni tipo di pericolo.

    Confermato quando, nella finestra:
      - la durata tra prima e ultima osservazione positiva è >= min_span;
      - le osservazioni positive sono almeno min_hits;
      - la frazione di frame positivi (dal primo avvistamento) è >= min_ratio.
    La confidenza restituita è la media degli score positivi nella finestra.
    """

    def __init__(self, window=2.0, min_span=1.0, min_hits=6, min_ratio=0.5):
        self.window = window
        self.min_span = min_span
        self.min_hits = max(1, min_hits)
        self.min_ratio = min_ratio
        self._obs = {}   # danger_type -> deque[(t, score)]

    def reset(self):
        self._obs.clear()

    def update(self, now, dangers):
        """
        dangers: {danger_type: score migliore nel frame} (solo i pericoli visti ora).
        Ritorna {danger_type: confidenza media} dei pericoli confermati in questo frame.
        """
        for danger_type in set(self._obs) | set(dangers):
            obs = self._obs.setdefault(danger_type, deque())
            obs.append((now, dangers.get(danger_type, 0.0)))
            while obs and now - obs[0][0] > self.window:
                obs.popleft()
            # Le osservazioni prima del primo avvistamento non contano nella frazione
            while obs and obs[0][1] <= 0.0:
                obs.popleft()
            if not obs:
                del self._obs[danger_type]

        confirmed = {}
        for danger_type, obs in self._obs.items():
            stats = self._stats(obs)
            if stats and stats[0] >= self.min_span and stats[1] >= self.min_hits and stats[2] >= self.min_ratio:
                confirmed[danger_type] = stats[3]
        return confirmed

    def progress(self):
        """{danger_type: avanzamento 0..1 verso la conferma}, per l'HUD."""
        out = {}
        for danger_type, obs in self._obs.items():
            stats = self._stats(obs)
            if stats:
                span, hits, ratio, _ = stats
                out[danger_type] = max(0.0, min(1.0, span / self.min_span if self.min_span > 0 else 1.0,
                                                hits / self.min_hits,
                                                ratio / self.min_ratio if self.min_ratio > 0 else 1.0))
        return out

    @staticmethod
    def _stats(obs):
        positives = [(t, s) for t, s in obs if s > 0.0]
        if not positives:
            return None
        span = positives[-1][0] - positives[0][0]
        hits = len(positives)
        ratio = hits / len(obs)
        mean_conf = sum(s for _, s in positives) / hits
        return span, hits, ratio, mean_conf


# Indici COCO-pose dei polsi (0-based): 9 = polso sinistro, 10 = polso destro
WRIST_KEYPOINTS = (9, 10)


def _point_box_distance(px, py, box):
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return math.hypot(dx, dy)


def near_hand(box, persons, margin_ratio=0.25, kp_conf=0.3):
    """
    True se il box dell'oggetto è vicino a un polso di una persona (modello pose).
    Margine = il maggiore tra mezza diagonale dell'oggetto e margin_ratio
    dell'altezza della persona: un coltello impugnato sta in mano, non lontano.
    """
    bw, bh = box[2] - box[0], box[3] - box[1]
    obj_margin = 0.5 * math.hypot(bw, bh)
    for person in persons:
        kps = person.get("keypoints")
        if kps is None:
            continue
        pb = person["box"]
        margin = max(obj_margin, margin_ratio * (pb[3] - pb[1]))
        for k in WRIST_KEYPOINTS:
            if k >= len(kps):
                continue
            kx, ky, kc = kps[k][0], kps[k][1], kps[k][2]
            if kc >= kp_conf and _point_box_distance(float(kx), float(ky), box) <= margin:
                return True
    return False


def threshold_for(base, near, hand_bonus):
    """Soglia effettiva: più bassa se l'oggetto è in mano a qualcuno."""
    return max(0.05, base - hand_bonus) if near else base
