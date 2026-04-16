# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

## Running the project

**Play the game (keyboard fallback if no model exists):**
```bash
python ejemplo_juego.py
```

**Collect training data (requires a webcam):**
```bash
python training/collector.py
```
Follow the prompts: for each of the 4 directions, position your index finger and press SPACE to capture 60 samples.

**Train the classifier:**
```bash
python training/trainer.py
```
Reads `training/data/X.npy` and `training/data/y.npy`, outputs `training/model.pkl`.

## Architecture

The project is a Pygame game controlled by hand gestures via computer vision. There are three independent layers:

**`vision/`** — camera and preprocessing
- `camera.py`: `CameraThread` captures frames in a background thread (non-blocking).
- `hand_segmenter.py`: `HandSegmenter` uses MediaPipe Hands to detect 21 hand landmarks and returns a (42,) float vector (x,y per landmark), normalized so the wrist is the origin and scale is relative to the wrist-to-middle-MCP distance. This makes features invariant to hand position, size, and background.
- `predictor.py`: `DirectionPredictor` loads `training/model.pkl` and runs the StandardScaler → PCA(20) → LogisticRegression pipeline to map the 42-vector to one of `{up, down, left, right}`.

**`training/`** — data collection and model training
- `collector.py`: Interactive OpenCV script that captures labeled samples for each direction and saves `training/data/X.npy` / `training/data/y.npy`.
- `trainer.py`: Trains a `StandardScaler → PCA(30 components) → LogisticRegression` pipeline and serializes it as a single `model.pkl` dict with keys `scaler`, `pca`, `model`.

**`game/`** — Pygame game logic
- `settings.py`: All magic numbers (window size, colors, speed, FPS).
- `controller.py`: Pure functions `apply_movement` and `wrap_position` — no Pygame dependency, easy to unit test.
- `renderer.py`: `draw_wrapped_rect` handles the screen-wrap visual effect by drawing up to 4 copies of the rectangle near boundaries.

**`ejemplo_juego.py`** — main entry point. Attempts to import the vision modules; if the model is missing or imports fail it falls back silently to keyboard control (arrow keys).

## Key design decisions

- The vision pipeline is **optional**: the game runs without a trained model or camera.
- `wrap_position` resets coordinates only after the rectangle has fully exited the screen, while `draw_wrapped_rect` draws the partial overlap on the opposite edge — these two functions must stay in sync.
- Hand detection uses MediaPipe Hands (`min_detection_confidence=0.7`). If detection is unreliable, adjust those thresholds in `vision/hand_segmenter.py`.
