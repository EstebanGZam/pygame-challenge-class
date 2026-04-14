import cv2
import numpy as np

IMG_SIZE = 50
HSV_LOWER = np.array([0, 30, 60])
HSV_UPPER = np.array([20, 150, 255])
MIN_HAND_PIXELS = 3000
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class HandSegmenter:
    """
    Isolates the hand from the background using HSV skin-color segmentation.
    Plays the same role as Otsu binarization in the lab notebook: a deterministic
    preprocessing step that removes the background before the classifier sees the data.
    """

    def get_mask(self, frame):
        """Returns the raw HSV skin mask (used for live preview in the collector)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL)
        return mask

    def segment(self, frame):
        """
        Returns a (2500,) vector with the binarized hand silhouette resized to
        50x50, or None if not enough skin pixels are detected.
        """
        mask = self.get_mask(frame)
        if np.count_nonzero(mask) < MIN_HAND_PIXELS:
            return None
        resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        return (resized > 127).astype(float).flatten()
