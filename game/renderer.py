import pygame


def draw_wrapped_rect(surface, color, rx, ry, rw, rh, W, H):
    """
    Draws the rectangle and, if it is crossing a boundary, draws the
    overlapping portion on the opposite side (gradual wrap effect).
    Corner cases for diagonal movement are also handled.
    """
    pygame.draw.rect(surface, color, (rx, ry, rw, rh))

    dx = W if rx < 0 else (-W if rx + rw > W else 0)
    dy = H if ry < 0 else (-H if ry + rh > H else 0)

    if dx:
        pygame.draw.rect(surface, color, (rx + dx, ry, rw, rh))
    if dy:
        pygame.draw.rect(surface, color, (rx, ry + dy, rw, rh))
    if dx and dy:
        pygame.draw.rect(surface, color, (rx + dx, ry + dy, rw, rh))
