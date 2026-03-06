import math

def get_crosshair_position(frame):

    height, width, _ = frame.shape

    cx = width // 2
    cy = height // 2

    return cx, cy


def calculate_distance(cx, cy, ex, ey):

    distance = math.sqrt((ex - cx)**2 + (ey - cy)**2)

    return distance