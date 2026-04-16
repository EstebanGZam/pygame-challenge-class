import cv2
import numpy as np
from mediapipe.python.solutions import hands as _mp_hands
from mediapipe.python.solutions import drawing_utils as _mp_drawing


class HandSegmenter:
    """
    Detects the hand using MediaPipe Hands and extracts 21 landmark coordinates
    as a normalized feature vector. This replaces the previous HSV skin-color
    approach, which was sensitive to background color and lighting conditions.

    The output vector is (42,): x,y for each of the 21 landmarks, translated
    so the wrist is at the origin and scaled by the wrist-to-middle-MCP distance.
    This makes the features invariant to hand position, size, and background.
    """

    def __init__(self):
        self._hands = _mp_hands.Hands(  # type: ignore[attr-defined]
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

    def _detect(self, frame):
        """Returns raw MediaPipe landmarks for the first hand found, or None."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None

    def draw(self, frame):
        """Returns a copy of the frame with hand landmarks drawn (used for collector preview)."""
        landmarks = self._detect(frame)
        out = frame.copy()
        if landmarks:
            _mp_drawing.draw_landmarks(out, landmarks, _mp_hands.HAND_CONNECTIONS)  # type: ignore[attr-defined]
        return out

    def segment(self, frame):
        """
        Returns a (42,) float vector of normalized landmark coordinates,
        or None if no hand is detected.

        Normalization: wrist (landmark 0) is translated to origin; coordinates
        are divided by the wrist-to-middle-finger-MCP (landmark 9) distance so
        the vector is scale-invariant.
        """
        landmarks = self._detect(frame)
        if landmarks is None:
            return None

        coords = np.array([[lm.x, lm.y] for lm in landmarks.landmark], dtype=float)  # (21, 2)

        # Translate so wrist is at origin
        coords -= coords[0]

        # Scale by wrist-to-middle-MCP distance
        scale = np.linalg.norm(coords[9])
        if scale < 1e-6:
            return None
        coords /= scale

        return coords.flatten()  # (42,)
