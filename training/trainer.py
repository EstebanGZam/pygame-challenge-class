"""
Trains the StandardScaler -> PCA -> LogisticRegression pipeline on the data
collected with collector.py and saves the model to training/model.pkl.
Usage: python training/trainer.py
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = Path(__file__).parent / 'data'
MODEL_PATH = Path(__file__).parent / 'model.pkl'
DIRECTIONS = ['up', 'down', 'left', 'right']
N_COMPONENTS = 20  # input is now 42 features (21 landmarks × x,y)


def train():
    X = np.load(DATA_DIR / 'X.npy')
    y = np.load(DATA_DIR / 'y.npy')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f'Explained variance ({N_COMPONENTS} components): {explained:.1f}%')

    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(X_train_p, y_train)

    y_pred = model.predict(X_test_p)
    print(f'Test accuracy: {accuracy_score(y_test, y_pred) * 100:.1f}%')
    print(classification_report(y_test, y_pred, target_names=DIRECTIONS))

    joblib.dump({'scaler': scaler, 'pca': pca, 'model': model}, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
    train()
