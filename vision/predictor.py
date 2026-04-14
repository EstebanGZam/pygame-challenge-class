import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / 'training' / 'model.pkl'
DIRECTION_MAP = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}


class DirectionPredictor:
    """
    Loads the trained pipeline (StandardScaler -> PCA -> LogisticRegression)
    and predicts finger direction from the HSV segmentation vector.
    """

    def __init__(self):
        self._scaler = None
        self._pca = None
        self._model = None
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            artifacts = joblib.load(MODEL_PATH)
            self._scaler = artifacts['scaler']
            self._pca = artifacts['pca']
            self._model = artifacts['model']
        else:
            print(f'[predictor] Model not found at {MODEL_PATH}')
            print('[predictor] Run training/collector.py and training/trainer.py first.')

    def is_ready(self):
        return self._model is not None

    def predict(self, vector):
        """Receives a (2500,) vector and returns 'up', 'down', 'left' or 'right'."""
        if not self.is_ready():
            return None
        x = self._scaler.transform(vector.reshape(1, -1))
        x = self._pca.transform(x)
        return DIRECTION_MAP[self._model.predict(x)[0]]
