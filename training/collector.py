"""
Collects training samples for the direction classifier.
Usage: python training/collector.py

For each direction the script shows a live camera feed alongside the HSV mask
so you can verify that skin segmentation is working before capturing samples.
Press SPACE to start capturing each class.
"""
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from vision.hand_segmenter import HandSegmenter

DATA_DIR = Path(__file__).parent / 'data'
DIRECTIONS = ['up', 'down', 'left', 'right']
SAMPLES_PER_CLASS = 60


def _preview_until_space(cap, segmenter, window):
    print('  Press SPACE to start capturing...', flush=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        mask = segmenter.get_mask(frame)
        side_by_side = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(window, side_by_side)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break


def _capture_samples(cap, segmenter, window):
    samples, count = [], 0
    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            continue
        mask = segmenter.get_mask(frame)
        side_by_side = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(window, side_by_side)
        cv2.waitKey(1)
        vector = segmenter.segment(frame)
        if vector is not None:
            samples.append(vector)
            count += 1
            print(f'  {count}/{SAMPLES_PER_CLASS}', end='\r', flush=True)
    print()
    return samples


def collect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    segmenter = HandSegmenter()
    cap = cv2.VideoCapture(0)
    win = 'Left: camera | Right: HSV mask'

    X, y = [], []
    for label, direction in enumerate(DIRECTIONS):
        print(f'\n[{label + 1}/4] Point your index finger: {direction.upper()}')
        _preview_until_space(cap, segmenter, win)
        samples = _capture_samples(cap, segmenter, win)
        X.extend(samples)
        y.extend([label] * len(samples))

    cap.release()
    cv2.destroyAllWindows()

    np.save(DATA_DIR / 'X.npy', np.array(X))
    np.save(DATA_DIR / 'y.npy', np.array(y))
    print(f'\nData saved to {DATA_DIR}')
    print(f'Total: {len(y)} samples ({SAMPLES_PER_CLASS} per class)')


if __name__ == '__main__':
    collect()
