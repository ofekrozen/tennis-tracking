from .utils import is_point_in_polygon, line_segment_intersection # Ensure line_segment_intersection is imported
import numpy as np

# Placeholder for net height in pixels (this would ideally be calculated or calibrated)
NET_HEIGHT_PIXELS = 20 # This is a very rough guess and needs proper handling
# Define how close the ball needs to be to the net to be considered for a net miss
NET_MISS_PROXIMITY_THRESHOLD = 10 # pixels

def detect_net_misses(ball_trajectories, net_line_coords):
    """
    Detects if shots hit the net and do not go over.
    This is a simplified 2D implementation.

    Args:
        ball_trajectories (list of list of tuples):
            Each inner list represents a shot's trajectory, containing (x, y, frame_num) tuples.
            For initial integration, this might be a single list representing the whole ball path.
        net_line_coords (tuple of four floats or two tuples):
            (x1, y1, x2, y2) or ((x1,y1), (x2,y2)) representing the net line in image coordinates.

    Returns:
        int: Count of net misses.
    """
    net_miss_count = 0
    if not net_line_coords:
        print("Warning: Net line coordinates are invalid for net miss detection.")
        return 0

    if len(net_line_coords) == 4: # (x1,y1,x2,y2)
        net_p1 = (net_line_coords[0], net_line_coords[1])
        net_p2 = (net_line_coords[2], net_line_coords[3])
    elif len(net_line_coords) == 2: # ((x1,y1), (x2,y2))
        net_p1 = net_line_coords[0]
        net_p2 = net_line_coords[1]
    else:
        print("Warning: Net line coordinates format not recognized.")
        return 0

    for trajectory in ball_trajectories: # Expects a list of trajectories (shots)
        if not trajectory or len(trajectory) < 2:
            continue

        # For this simplified version, we assume each 'trajectory' is a single shot.
        # A more advanced version would get actual shots from rally_stats.py.

        previous_point = trajectory[0][:2] # (x,y)
        for current_frame_idx in range(1, len(trajectory)):
            current_point = trajectory[current_frame_idx][:2]
            ball_segment_p1 = previous_point
            ball_segment_p2 = current_point

            intersection_point = line_segment_intersection(ball_segment_p1, ball_segment_p2, net_p1, net_p2)

            if intersection_point is not None:
                # Intersection detected. Now, a heuristic to see if it's a "miss".
                # Simplification: if after this intersection, the ball doesn't clearly continue
                # far beyond the net, or if its y-coordinate relative to net suggests it didn't go over.
                # This needs to be more robust.

                # Check if the ball stops or "bounces back" immediately after intersection.
                # Look at a few points after the intersection.
                points_after_intersection = trajectory[current_frame_idx:]

                if len(points_after_intersection) > 1:
                    # Calculate average position of ball after intersection for a few frames
                    avg_pos_after_net = np.mean([p[:2] for p in points_after_intersection[:5]], axis=0)

                    # Simple check: If the average Y after intersection is still very close to net's Y,
                    # or on the same side as before, it might be a net miss.
                    # This also doesn't account for ball going over and dropping.
                    # A proper check needs 3D or better side-of-net logic.

                    # Let's use a simpler heuristic for now: if the ball's path ends near the net after intersection.
                    dist_to_net_at_end = np.linalg.norm(np.array(trajectory[-1][:2]) - np.array(intersection_point))

                    if dist_to_net_at_end < NET_MISS_PROXIMITY_THRESHOLD * 5: # If ball ends up close to intersection point
                         # Further check: did it cross to the other side of the net?
                        p_before_net = trajectory[current_frame_idx-1][:2]
                        p_after_net = None
                        if current_frame_idx + 1 < len(trajectory):
                            p_after_net = trajectory[current_frame_idx+1][:2]

                        if p_after_net:
                            # A simple check based on y-coordinate relative to net line's average y.
                            # This assumes net is mostly horizontal in image.
                            net_avg_y = (net_p1[1] + net_p2[1]) / 2
                            if (p_before_net[1] < net_avg_y and p_after_net[1] < net_avg_y + NET_MISS_PROXIMITY_THRESHOLD) or \
                                (p_before_net[1] > net_avg_y and p_after_net[1] > net_avg_y - NET_MISS_PROXIMITY_THRESHOLD):
                                # Ball seems to have stayed on the same side or very close after hitting net
                                net_miss_count += 1
                                break # Count one net miss per trajectory (shot)
                elif len(points_after_intersection) <=1: # Trajectory ends at/immediately after net
                    net_miss_count += 1
                    break

            previous_point = current_point
    return net_miss_count

def detect_out_misses(ball_landing_positions, court_boundary_points):
    """
    Detects if the ball lands outside the court boundaries.

    Args:
        ball_landing_positions (list of tuples):
            List of (x, y) coordinates for detected ball bounces.
        court_boundary_points (list of tuples):
            Vertices of the court polygon [(x1,y1), (x2,y2), ...].

    Returns:
        int: Count of out misses.
    """
    out_miss_count = 0
    if not court_boundary_points or len(court_boundary_points) < 3:
        print("Warning: Court boundary points are invalid.")
        return 0

    for landing_pos in ball_landing_positions:
        if not is_point_in_polygon(landing_pos, court_boundary_points):
            out_miss_count += 1
    return out_miss_count

# In statistics_extraction/shot_accuracy_stats.py
# from .utils import is_point_in_polygon, line_segment_intersection # Ensure this is at the top

def check_serve_accuracy(serves_info): # Removed court_lines from args for now
    """
    Checks if serves are in or out based on pre-determined target service boxes.

    Args:
        serves_info (list of dicts): Each dict must contain:
            'landing_coord': (x, y) of the serve's bounce.
            'is_first_serve': boolean.
            'target_service_box_transformed_points': list of (x,y) tuples defining the target box
                                                      already transformed to image coordinates.
            # Optional: 'frame_of_bounce'
    Returns:
        tuple: (first_serves_in, first_serves_out, second_serves_in, second_serves_out)
    """
    first_serves_in = 0
    first_serves_out = 0
    second_serves_in = 0
    second_serves_out = 0

    for serve in serves_info:
        landing_coord = serve.get('landing_coord')
        is_first = serve.get('is_first_serve', True)
        target_box_points = serve.get('target_service_box_transformed_points')

        if not landing_coord or not target_box_points or len(target_box_points) < 3:
            print(f"Warning: Invalid serve info or target box for a serve: {serve.get('frame_of_bounce', 'N/A')}")
            if is_first:
                first_serves_out +=1
            else:
                second_serves_out += 1
            continue

        if is_point_in_polygon(landing_coord, target_box_points):
            if is_first:
                first_serves_in += 1
            else:
                second_serves_in += 1
        else:
            # Could add a check here for "net cord" serves if net intersection is also passed
            # For now, any non-in serve is out/fault.
            if is_first:
                first_serves_out += 1
            else:
                second_serves_out += 1

    return first_serves_in, first_serves_out, second_serves_in, second_serves_out

def detect_aces(serves_info, receiver_trajectories, ball_trajectories):
    """
    Detects aces. An ace is a legal serve that is not touched by the receiver.

    Args:
        serves_info (list of dicts): From `check_serve_accuracy`, including whether serve was 'in'.
            Each dict should have: 'landing_coord', 'is_in', 'frame_serve_ends'.
        receiver_trajectories (dict): Player ID -> list of (x,y,frame_num) for receiver.
        ball_trajectories (list of list of tuples): Ball trajectories for each serve.

    Returns:
        int: Count of aces.
    """
    ace_count = 0
    # This requires knowing which serve was in, who the receiver was,
    # and if the receiver made any significant movement to play the ball
    # or if the ball was touched (requires more advanced contact detection).

    # Simplified logic: If serve is IN, and receiver doesn't get close to the ball's path
    # after the bounce and before a second bounce (or point ends).

    for serve_idx, serve in enumerate(serves_info):
        if serve.get('is_in'):
            # Determine the receiver (needs to be passed in or inferred)
            receiver_id = serve.get('receiver_id')
            if not receiver_id or receiver_id not in receiver_trajectories:
                continue

            serve_ball_trajectory = ball_trajectories[serve_idx] # Assuming a 1-to-1 mapping

            # Check if receiver touched the ball. This is the hardest part.
            # For now, let's assume if the receiver doesn't get within a certain distance
            # of the ball's trajectory after the bounce, it's an ace.

            # Example: Find ball position at serve bounce.
            # Track receiver's position from that frame onwards for a short duration.
            # If receiver remains far from the ball's path, it's an ace.
            # This is a placeholder for more sophisticated logic.
            pass
    return ace_count
