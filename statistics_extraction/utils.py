import numpy as np

def is_point_in_polygon(point, polygon_points):
    """
    Checks if a 2D point is inside a 2D polygon.

    Args:
        point (tuple or list): The (x, y) coordinates of the point.
        polygon_points (list of tuples): A list of (x, y) coordinates representing the vertices of the polygon,
                                         in order (clockwise or counter-clockwise).

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    x, y = point
    n = len(polygon_points)
    inside = False

    p1x, p1y = polygon_points[0]
    for i in range(n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

# Example Usage (can be removed or kept for testing):
if __name__ == '__main__':
    # Simple square
    square = [(0, 0), (0, 2), (2, 2), (2, 0)]
    point_inside = (1, 1)
    point_outside = (3, 3)
    point_on_edge = (0, 1) # Behavior for points on edge can vary; this algorithm might count them as inside or outside.

    print(f"Point {point_inside} in square: {is_point_in_polygon(point_inside, square)}") # Expected: True
    print(f"Point {point_outside} in square: {is_point_in_polygon(point_outside, square)}") # Expected: False
    print(f"Point {point_on_edge} in square: {is_point_in_polygon(point_on_edge, square)}") # Expected: True or False (often True)

    # Triangle
    triangle = [(0,0), (5,0), (2.5,5)]
    point_inside_triangle = (2.5, 2)
    point_outside_triangle = (0,1)
    print(f"Point {point_inside_triangle} in triangle: {is_point_in_polygon(point_inside_triangle, triangle)}") # Expected: True
    print(f"Point {point_outside_triangle} in triangle: {is_point_in_polygon(point_outside_triangle, triangle)}") # Expected: False

def line_segment_intersection(p1, p2, p3, p4):
    """
    Checks if two line segments, (p1, p2) and (p3, p4), intersect.
    Returns the intersection point or None.
    p1, p2, p3, p4 are all (x, y) tuples.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None  # Parallel or collinear

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))

    t = t_num / den
    u = u_num / den

    if 0 <= t <= 1 and 0 <= u <= 1:
        # Intersection point
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    return None # No intersection within segments
