import cv2
import threading


class CameraThread:
    """
    Captures frames from the camera in a background thread so the
    main game loop is never blocked waiting for a frame.
    """

    def __init__(self, device=0):
        self._cap = cv2.VideoCapture(device)
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False
        self._cap.release()

    def get_frame(self):
        """Returns the latest captured frame, or None if none is available yet."""
        with self._lock:
            return self._frame

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
