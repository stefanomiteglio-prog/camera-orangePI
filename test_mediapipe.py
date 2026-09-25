import mediapipe as mp
hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
print('OK, mediapipe hands inizializzato')
