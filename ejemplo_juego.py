import pygame
import sys
from game.settings import (
    WIDTH, HEIGHT, LIGHT_GREEN, BLUE, FPS,
    RECT_WIDTH, RECT_HEIGHT, RECT_X_INIT, RECT_Y_INIT, SPEED,
)
from game.renderer import draw_wrapped_rect
from game.controller import apply_movement, wrap_position

# --- Vision module initialization (optional) ---
USE_CAMERA = False
cam = None
segmenter = None
predictor = None

try:
    from vision.camera import CameraThread
    from vision.hand_segmenter import HandSegmenter
    from vision.predictor import DirectionPredictor

    cam = CameraThread()
    segmenter = HandSegmenter()
    predictor = DirectionPredictor()

    if predictor.is_ready():
        cam.start()
        USE_CAMERA = True
        print('Camera mode activated.')
    else:
        print('Model not trained. Falling back to keyboard.')
except ImportError as e:
    print(f'Vision modules not available ({e}). Falling back to keyboard.')

# --- Pygame setup ---
pygame.init()
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Move rectangle')

rect_x, rect_y = RECT_X_INIT, RECT_Y_INIT
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if cam:
                cam.stop()
            pygame.quit()
            sys.exit()

    direction = None

    if USE_CAMERA:
        frame = cam.get_frame()
        if frame is not None:
            vector = segmenter.segment(frame)
            if vector is not None:
                direction = predictor.predict(vector)

    # Keyboard fallback if the camera provides no direction
    if direction is None:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            direction = 'left'
        elif keys[pygame.K_RIGHT]:
            direction = 'right'
        elif keys[pygame.K_UP]:
            direction = 'up'
        elif keys[pygame.K_DOWN]:
            direction = 'down'

    rect_x, rect_y = apply_movement(direction, rect_x, rect_y, SPEED)
    rect_x, rect_y = wrap_position(rect_x, rect_y, RECT_WIDTH, RECT_HEIGHT, WIDTH, HEIGHT)

    window.fill(LIGHT_GREEN)
    draw_wrapped_rect(window, BLUE, rect_x, rect_y, RECT_WIDTH, RECT_HEIGHT, WIDTH, HEIGHT)
    pygame.display.flip()
    clock.tick(FPS)
