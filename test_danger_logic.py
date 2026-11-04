"""
Test della logica di conferma armi (danger_logic.py), senza NPU né telecamera.

    python test_danger_logic.py
"""
import unittest

from danger_logic import DangerConfirmer, near_hand, threshold_for

FPS = 18.0


def run(confirmer, pattern, seconds, score=0.6, danger="coltello", start=0.0):
    """pattern(i) -> True se il coltello è visto al frame i. Ritorna il primo istante di conferma."""
    t = start
    for i in range(int(seconds * FPS)):
        t = start + i / FPS
        confirmed = confirmer.update(t, {danger: score} if pattern(i) else {})
        if danger in confirmed:
            return t, confirmed[danger]
    return None, None


class DangerConfirmerTest(unittest.TestCase):
    def make(self):
        return DangerConfirmer(window=2.0, min_span=1.0, min_hits=6, min_ratio=0.5)

    def test_steady_detection_confirms_after_min_span(self):
        t, conf = run(self.make(), lambda i: True, 3.0, score=0.7)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 1.0, delta=0.1)
        self.assertAlmostEqual(conf, 0.7, places=5)

    def test_flickering_detection_still_confirms(self):
        # Visto 2 frame su 3 (detector che sfarfalla)
        t, _ = run(self.make(), lambda i: i % 3 != 0, 3.0)
        self.assertIsNotNone(t)

    def test_gap_does_not_reset(self):
        # 0.6 s visto, 0.5 s perso (lama di taglio), poi di nuovo visto
        pattern = lambda i: not (11 <= i < 20)
        t, _ = run(self.make(), pattern, 3.0)
        self.assertIsNotNone(t)
        self.assertLess(t, 1.6)

    def test_sporadic_false_positive_never_confirms(self):
        # Un frame ogni 10: tipico falso positivo isolato
        t, _ = run(self.make(), lambda i: i % 10 == 0, 10.0)
        self.assertIsNone(t)

    def test_short_burst_never_confirms(self):
        # 0.5 s di detection poi più niente
        t, _ = run(self.make(), lambda i: i < 9, 5.0)
        self.assertIsNone(t)

    def test_state_expires(self):
        c = self.make()
        run(c, lambda i: i < 9, 0.5)
        run(c, lambda i: False, 3.0, start=0.5)
        self.assertEqual(c.progress(), {})

    def test_progress_grows(self):
        c = self.make()
        c.update(0.0, {"coltello": 0.5})
        p1 = c.progress()["coltello"]
        for i in range(1, 9):
            c.update(i / FPS, {"coltello": 0.5})
        self.assertGreater(c.progress()["coltello"], p1)


class NearHandTest(unittest.TestCase):
    person = {"box": [100, 50, 300, 450],
              "keypoints": [[0, 0, 0]] * 9 + [[150, 260, 0.9], [260, 270, 0.9]] + [[0, 0, 0]] * 6}

    def test_knife_in_hand(self):
        self.assertTrue(near_hand([255, 230, 275, 320], [self.person]))

    def test_knife_far_from_hands(self):
        self.assertFalse(near_hand([500, 400, 520, 470], [self.person]))

    def test_low_confidence_wrist_ignored(self):
        p = dict(self.person, keypoints=[[0, 0, 0]] * 9 + [[150, 260, 0.1], [260, 270, 0.1]] + [[0, 0, 0]] * 6)
        self.assertFalse(near_hand([255, 230, 275, 320], [p]))

    def test_threshold(self):
        self.assertAlmostEqual(threshold_for(0.4, True, 0.1), 0.3)
        self.assertAlmostEqual(threshold_for(0.4, False, 0.1), 0.4)


if __name__ == "__main__":
    unittest.main()
