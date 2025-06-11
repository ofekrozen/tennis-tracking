import numpy as np
from .shot_accuracy_stats import detect_out_misses, detect_net_misses

MIN_SHOT_SPEED_CHANGE_THRESHOLD = 10  # Example threshold for detecting a hit (pixels/frame)
MIN_RALLY_SHOTS = 1 # A rally must have at least one shot (the serve)

# In identify_rallies_and_shots function in statistics_extraction/rally_stats.py

# ... (imports and _segment_into_shots function remain the same) ...

def identify_rallies_and_shots(ball_trajectory, player_positions, court_info,
                                   net_line_coords, court_boundary_points,
                                   all_detected_bounces=None): # New arg: list of (x,y,frame_num)
    """
    Identifies individual shots and groups them into rallies.
    A rally ends if a shot is a net miss (stopping play) or an actual detected bounce lands out.

    Args:
        ball_trajectory (list of tuples):
            Contains (x, y, frame_num, [optional_z]) for the ball throughout a play segment.
            Assumes coordinates are cleaned and possibly interpolated.
        player_positions (dict):
            Maps player_id to a list of their (x, y, frame_num) positions.
        court_info (object or dict):
            Contains court boundary lines, net position, service lines.
            (e.g., from court_detector.lines and court_detector.court_reference)
        net_line_coords (tuple): (x1,y1,x2,y2) for the net. Used for net miss detection.
        court_boundary_points (list of tuples): Vertices of the court. Used for out miss detection.


    Returns:
        list of list of dicts:
            A list of rallies. Each rally is a list of shot dictionaries.
            Each shot dictionary could contain: {
                'start_frame': int,
                'end_frame': int,
                'ball_path': list_of_coords,
                'player_id': str_or_int (optional, if identifiable)
            }
    """
    rallies = []
    if not ball_trajectory:
        return rallies

    # This is a complex task. Steps involved:
    # 1. Detect Serves: Identify the start of each point/rally.
    #    - Look for ball starting near baseline, characteristic motion.
    #    - This might need specific logic or flags from the main processing loop.

    # 2. Segment Ball Trajectory into Shots:
    #    - Calculate ball velocity between consecutive points.
    #    - A significant change in velocity (speed or direction) indicates a hit.
    #    - Player proximity at the time of hit can confirm it's a player shot.

    # 3. Group Shots into Rallies:
    #    - A rally starts with a serve (or first detected shot).
    #    - A rally ends when:
    #        - A net miss is detected (using functions from shot_accuracy_stats).
    #        - An out miss (ball landing out) is detected.
    #        - Ball bounces twice (hard to detect reliably without good Z-axis).
    #        - Ball comes to a stop or is no longer in play.
    if all_detected_bounces is None:
        all_detected_bounces = []

    individual_shots = _segment_into_shots(ball_trajectory, player_positions)

    if not individual_shots:
        return rallies

    current_rally_shots = []
    for shot_idx, shot in enumerate(individual_shots):
        current_rally_shots.append(shot)

        # Determine if this shot ends the rally
        shot_ends_rally = False

        # 1. Check for Net Miss (that stops play)
        # We pass only the current shot's trajectory to detect_net_misses
        if net_line_coords and detect_net_misses([shot['ball_path']], net_line_coords) > 0:
            # Assuming detect_net_misses correctly identifies net shots that would stop play
            shot_ends_rally = True
            # print(f"DEBUG Rally End: Net miss on shot starting frame {shot['start_frame']}")

        # 2. Check for Out Miss (ball landing out) - only if not already ended by net
        if not shot_ends_rally and court_boundary_points:
            # detect_out_misses expects a list of landing positions.
            # For a single shot, we take its last point as the potential landing position.
            # This is a simplification; true bounce detection for the shot would be better.
            if shot['ball_path']:
                # Use the last point of the ball path for out detection.
                # A more robust way would be to find the actual bounce point of this shot.
                potential_landing_point = shot['ball_path'][-1][:2] # (x,y)
                if detect_out_misses([potential_landing_point], court_boundary_points) > 0:
                    shot_ends_rally = True
                    # print(f"DEBUG Rally End: Out miss on shot starting frame {shot['start_frame']}")

        # If the shot ends the rally, finalize the current rally and start a new one
        if shot_ends_rally:
            if len(current_rally_shots) >= MIN_RALLY_SHOTS:
                rallies.append(list(current_rally_shots))
            current_rally_shots = [] # Start a new rally
        # If it's the last shot overall and hasn't ended a rally, it completes the current rally
        elif shot_idx == len(individual_shots) - 1:
             if len(current_rally_shots) >= MIN_RALLY_SHOTS:
                rallies.append(list(current_rally_shots))
                current_rally_shots = [] # Clear it after adding

    # If there are any remaining shots in current_rally_shots (e.g. last shot didn't end rally explicitly but loop finished)
    # This case should be covered by the "elif shot_idx == len(individual_shots) - 1"
    # but as a safeguard:
    if len(current_rally_shots) >= MIN_RALLY_SHOTS:
        rallies.append(list(current_rally_shots))

    return rallies


def _segment_into_shots(ball_trajectory, player_positions=None,
                            min_angle_change_deg=30, min_speed_change_ratio=0.5,
                            min_shot_duration_frames=3, min_ball_speed_pixels_per_frame=2):
    """
    Segments a continuous ball trajectory into discrete shots based on changes in velocity.

    Args:
        ball_trajectory (list of tuples): (x, y, frame_num, [optional_z])
        player_positions (dict): Optional. Maps player_id to their positions. Not used in this version.
        min_angle_change_deg (float): Minimum change in direction (degrees) to detect a hit.
        min_speed_change_ratio (float): Minimum ratio of speed change (abs(s2-s1)/max(s1,s2)) for a hit.
                                        Not fully implemented in this version, focusing on angle.
        min_shot_duration_frames (int): Minimum frames a shot should last.
        min_ball_speed_pixels_per_frame (float): Minimum speed for the ball to be considered actively in play for a shot.


    Returns:
        list of dicts: Each dict represents a shot with 'start_frame', 'end_frame', 'ball_path'.
    """
    detected_shots = []
    if len(ball_trajectory) < min_shot_duration_frames * 2: # Need enough points to form shots
        return detected_shots

    current_shot_points = []
    velocities = [] # List of (dx, dy, speed, angle_rad, frame_num)

    # Calculate velocities and angles
    for i in range(len(ball_trajectory) - 1):
        p1 = ball_trajectory[i]
        p2 = ball_trajectory[i+1]

        frame1, x1, y1 = p1[2], p1[0], p1[1]
        frame2, x2, y2 = p2[2], p2[0], p2[1]

        if frame2 == frame1: # Avoid division by zero if duplicate frames
            continue

        dx = (x2 - x1) / (frame2 - frame1)
        dy = (y2 - y1) / (frame2 - frame1)
        speed = np.sqrt(dx**2 + dy**2)
        angle_rad = np.arctan2(dy, dx)
        velocities.append({'dx': dx, 'dy': dy, 'speed': speed, 'angle': angle_rad, 'frame_num': frame1, 'original_point': p1})

    if not velocities:
        return detected_shots

    # Add the last point to complete the trajectory for the last velocity segment
    velocities.append({'dx': 0, 'dy': 0, 'speed': 0, 'angle': velocities[-1]['angle'],
                       'frame_num': ball_trajectory[-1][2], 'original_point': ball_trajectory[-1]})


    current_shot_start_velocity_idx = 0
    for i in range(1, len(velocities)):
        prev_vel = velocities[i-1]
        curr_vel = velocities[i]

        if prev_vel['speed'] < min_ball_speed_pixels_per_frame and current_shot_start_velocity_idx == i-1:
            # If ball is too slow and we haven't started a shot yet, advance start
            current_shot_start_velocity_idx = i
            continue

        angle_diff_rad = abs(curr_vel['angle'] - prev_vel['angle'])
        # Normalize angle difference to be between 0 and pi
        if angle_diff_rad > np.pi:
            angle_diff_rad = 2 * np.pi - angle_diff_rad
        angle_diff_deg = np.degrees(angle_diff_rad)

        # speed_ratio_diff = 0
        # if max(prev_vel['speed'], curr_vel['speed']) > 0.1: # Avoid division by zero for very slow speeds
        #     speed_ratio_diff = abs(curr_vel['speed'] - prev_vel['speed']) / max(prev_vel['speed'], curr_vel['speed'])

        # Primary condition for now: significant angle change OR ball comes to a near stop and restarts
        is_hit_boundary = angle_diff_deg > min_angle_change_deg

        # Condition for ball stopping then restarting (could be a new shot/serve)
        ball_restarted = prev_vel['speed'] < min_ball_speed_pixels_per_frame and \
                         curr_vel['speed'] >= min_ball_speed_pixels_per_frame


        if is_hit_boundary or ball_restarted or i == len(velocities) - 1: # End of trajectory also finalizes a shot
            shot_points_indices = range(current_shot_start_velocity_idx, i)

            # Ensure the shot has minimum duration and includes enough original points
            if len(shot_points_indices) >= min_shot_duration_frames:
                shot_ball_path = [velocities[j]['original_point'] for j in shot_points_indices]
                # Add the point that *caused* the boundary to delimit the end of this shot accurately
                shot_ball_path.append(velocities[i]['original_point'] if i < len(velocities) else velocities[i-1]['original_point'])


                if shot_ball_path:
                    start_frame = shot_ball_path[0][2]
                    end_frame = shot_ball_path[-1][2]

                    # Filter out very short "shots" that might be noise
                    if (end_frame - start_frame) >= min_shot_duration_frames -1 : # -1 because frames are inclusive start/end
                         # Further filter: ensure average speed of shot is reasonable
                        avg_speed_of_shot = np.mean([v['speed'] for v_idx, v in enumerate(velocities) if current_shot_start_velocity_idx <= v_idx < i])
                        if avg_speed_of_shot >= min_ball_speed_pixels_per_frame:
                            detected_shots.append({
                                'start_frame': start_frame,
                                'end_frame': end_frame,
                                'ball_path': shot_ball_path
                            })
            current_shot_start_velocity_idx = i

    return detected_shots


def calculate_longest_rally(rallies):
    """
    Calculates the length of the longest rally in terms of number of shots.

    Args:
        rallies (list of list of dicts): Output from identify_rallies_and_shots.

    Returns:
        int: Number of shots in the longest rally. Returns 0 if no rallies.
    """
    if not rallies:
        return 0

    max_shots = 0
    for rally in rallies:
        if len(rally) > max_shots:
            max_shots = len(rally)

    return max_shots

# TODO: Add functions related to serve detection if not handled elsewhere.
# def detect_serves(ball_trajectory, player_positions, court_info):
#     ...
#     return list_of_serve_events
