def apply_movement(direction, rx, ry, speed):
    """Applies the given direction to the rectangle and returns the new position."""
    if direction == 'left':
        rx -= speed
    elif direction == 'right':
        rx += speed
    if direction == 'up':
        ry -= speed
    elif direction == 'down':
        ry += speed
    return rx, ry


def wrap_position(rx, ry, rw, rh, W, H):
    """
    Resets the position only after the rectangle has fully crossed a boundary.
    While it is crossing, the position stays partially off-screen so that
    renderer.py can draw both halves simultaneously.
    """
    if rx >= W:
        rx -= W
    elif rx + rw <= 0:
        rx += W

    if ry >= H:
        ry -= H
    elif ry + rh <= 0:
        ry += H

    return rx, ry
