import numpy as np
from .shot_accuracy_stats import detect_out_misses, detect_net_misses

MIN_SHOT_SPEED_CHANGE_THRESHOLD = 10  # Example threshold for detecting a hit (pixels/frame)
MIN_RALLY_SHOTS = 1 # A rally must have at least one shot (the serve)

# In identify_rallies_and_shots function in statistics_extraction/rally_stats.py

# ... (imports and _segment_into_shots function remain the same) ...

def get_player_for_shot(shot_index_in_rally, server_id_for_rally):
    """Determines which player hit a shot based on serve alternation."""
    if server_id_for_rally not in ['player1', 'player2']:
        return "Unknown" # Or handle error

    # Server hits the 0th, 2nd, 4th... shot of the rally (0-indexed)
    if shot_index_in_rally % 2 == 0:
        return server_id_for_rally
    else:
        return 'player2' if server_id_for_rally == 'player1' else 'player1'

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
        all_detected_bounces (list of tuples): List of (x,y,frame_num) for all detected bounces.


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

    print(f"DEBUG_VALIDATE: identify_rallies_and_shots: Started. InputTrajPoints: {len(ball_trajectory)}, NumBounces: {len(all_detected_bounces if all_detected_bounces else [])}")
    if all_detected_bounces is None:
        all_detected_bounces = []

    individual_shots = _segment_into_shots(ball_trajectory, player_positions)
    print(f"DEBUG_VALIDATE: identify_rallies_and_shots: Found {len(individual_shots)} individual shots.")

    if not individual_shots:
        return rallies

    current_rally_shots = []
    for shot_idx, shot in enumerate(individual_shots):
        current_rally_shots.append(shot)

        shot_ends_rally = False
        shot_start_frame = shot['start_frame']
        shot_end_frame = shot['end_frame']

        if net_line_coords and shot.get('ball_path') and detect_net_misses([shot['ball_path']], net_line_coords) > 0:
            shot_ends_rally = True
            print(f"DEBUG_VALIDATE: identify_rallies_and_shots: Rally ended by NET MISS. ShotIdx: {shot_idx}, ShotFrames: {shot_start_frame}-{shot_end_frame}.")

        if not shot_ends_rally and court_boundary_points and shot.get('ball_path'):
            bounce_landed_out_for_this_shot = False
            for bounce_coord_x, bounce_coord_y, bounce_frame in all_detected_bounces:
                if shot_start_frame <= bounce_frame <= shot_end_frame + 5:
                    if detect_out_misses([(bounce_coord_x, bounce_coord_y)], court_boundary_points) > 0:
                        print(f"DEBUG_VALIDATE: identify_rallies_and_shots: Potential OUT bounce at frame {bounce_frame} (during shot {shot_idx}, frames {shot_start_frame}-{shot_end_frame}) determined rally end.")
                        bounce_landed_out_for_this_shot = True
                        break

            if bounce_landed_out_for_this_shot:
                shot_ends_rally = True

        if shot_ends_rally:
            if len(current_rally_shots) >= MIN_RALLY_SHOTS:
                rallies.append(list(current_rally_shots))
                if len(rallies) > 0:
                     rally_to_log = rallies[-1]
                     if rally_to_log:
                         print(f"DEBUG_VALIDATE: identify_rallies_and_shots: Finalized Rally #{len(rallies)}. Shots: {len(rally_to_log)}. RallyFrames: {rally_to_log[0]['start_frame']}-{rally_to_log[-1]['end_frame']}. Ended by this shot.")
            current_rally_shots = []
        elif shot_idx == len(individual_shots) - 1:
             if len(current_rally_shots) >= MIN_RALLY_SHOTS:
                rallies.append(list(current_rally_shots))
                if len(rallies) > 0:
                     rally_to_log = rallies[-1]
                     if rally_to_log:
                         print(f"DEBUG_VALIDATE: identify_rallies_and_shots: Finalized Rally #{len(rallies)} (last shot). Shots: {len(rally_to_log)}. RallyFrames: {rally_to_log[0]['start_frame']}-{rally_to_log[-1]['end_frame']}.")
                current_rally_shots = []

    if len(current_rally_shots) >= MIN_RALLY_SHOTS:
        rallies.append(list(current_rally_shots))
        if len(rallies) > 0:
             rally_to_log = rallies[-1]
             if rally_to_log:
                 print(f"DEBUG_VALIDATE: identify_rallies_and_shots: Finalized Rally #{len(rallies)} (safeguard). Shots: {len(rally_to_log)}. RallyFrames: {rally_to_log[0]['start_frame']}-{rally_to_log[-1]['end_frame']}.")

    return rallies


def _segment_into_shots(ball_trajectory, player_positions=None,
                            min_angle_change_deg=30, min_speed_change_ratio=0.5,
                            min_shot_duration_frames=3, min_ball_speed_pixels_per_frame=2):
    detected_shots = []
    if len(ball_trajectory) < min_shot_duration_frames * 2:
        return detected_shots

    velocities = []
    for i in range(len(ball_trajectory) - 1):
        p1 = ball_trajectory[i]
        p2 = ball_trajectory[i+1]
        frame1, x1, y1 = p1[2], p1[0], p1[1]
        frame2, x2, y2 = p2[2], p2[0], p2[1]
        if frame2 == frame1: continue
        dx = (x2 - x1) / (frame2 - frame1)
        dy = (y2 - y1) / (frame2 - frame1)
        speed = np.sqrt(dx**2 + dy**2)
        angle_rad = np.arctan2(dy, dx)
        velocities.append({'dx': dx, 'dy': dy, 'speed': speed, 'angle': angle_rad, 'frame_num': frame1, 'original_point': p1})

    if not velocities: return detected_shots

    velocities.append({'dx': 0, 'dy': 0, 'speed': 0, 'angle': velocities[-1]['angle'],
                       'frame_num': ball_trajectory[-1][2], 'original_point': ball_trajectory[-1]})

    current_shot_start_velocity_idx = 0
    for i in range(1, len(velocities)):
        prev_vel = velocities[i-1]
        curr_vel = velocities[i]

        if prev_vel['speed'] < min_ball_speed_pixels_per_frame and current_shot_start_velocity_idx == i-1:
            current_shot_start_velocity_idx = i
            continue

        angle_diff_rad = abs(curr_vel['angle'] - prev_vel['angle'])
        if angle_diff_rad > np.pi: angle_diff_rad = 2 * np.pi - angle_diff_rad
        angle_diff_deg = np.degrees(angle_diff_rad)

        is_hit_boundary = angle_diff_deg > min_angle_change_deg
        ball_restarted = prev_vel['speed'] < min_ball_speed_pixels_per_frame and \
                         curr_vel['speed'] >= min_ball_speed_pixels_per_frame

        if is_hit_boundary:
            print(f"DEBUG_VALIDATE: _segment_into_shots: Potential shot boundary at vel frame {i} (orig frame {curr_vel['frame_num']}). Reason: Angle change {angle_diff_deg:.2f} deg.")
        if ball_restarted:
            print(f"DEBUG_VALIDATE: _segment_into_shots: Potential shot boundary at vel frame {i} (orig frame {curr_vel['frame_num']}). Reason: Ball restarted (prev_speed={prev_vel['speed']:.2f}, curr_speed={curr_vel['speed']:.2f}).")

        if is_hit_boundary or ball_restarted or i == len(velocities) - 1:
            shot_points_indices = range(current_shot_start_velocity_idx, i)
            if len(shot_points_indices) >= min_shot_duration_frames:
                shot_ball_path = [velocities[j]['original_point'] for j in shot_points_indices]
                shot_ball_path.append(velocities[i]['original_point'] if i < len(velocities) else velocities[i-1]['original_point'])
                if shot_ball_path:
                    start_frame = shot_ball_path[0][2]
                    end_frame = shot_ball_path[-1][2]
                    if (end_frame - start_frame) >= min_shot_duration_frames -1 :
                        avg_speed_of_shot = np.mean([v['speed'] for v_idx, v in enumerate(velocities) if current_shot_start_velocity_idx <= v_idx < i and i > current_shot_start_velocity_idx]) # ensure i > start_idx for mean
                        if avg_speed_of_shot >= min_ball_speed_pixels_per_frame:
                            print(f"DEBUG_VALIDATE: _segment_into_shots: Finalized shot. StartFrame: {start_frame}, EndFrame: {end_frame}, NumPoints: {len(shot_ball_path)}, AvgSpeed: {avg_speed_of_shot:.2f}")
                            detected_shots.append({'start_frame': start_frame, 'end_frame': end_frame, 'ball_path': shot_ball_path})
            current_shot_start_velocity_idx = i
    return detected_shots

def calculate_longest_rally(rallies):
    if not rallies: return 0
    max_shots = 0
    for rally in rallies:
        if len(rally) > max_shots: max_shots = len(rally)
    return max_shots

def analyze_rally_outcomes(rallies_data, serves_info_list, court_boundary_points, net_line_coords):
    """
    Analyzes each rally to determine a tentative outcome, including attributing
    the last action to a player if possible.
    """
    rally_outcomes = []

    for rally_idx, rally_shots in enumerate(rallies_data):
        current_point_num = rally_idx

        outcome_details = {
            'rally_idx': current_point_num, # rally_idx is effectively the point number in this context
            'num_shots': 0,
            'hitter_of_last_shot': 'Unknown',
            'reason': 'Unknown',
            'type': 'Unknown'
        }

        if not rally_shots:
            outcome_details['reason'] = 'Empty Rally data'
            rally_outcomes.append(outcome_details)
            continue

        outcome_details['num_shots'] = len(rally_shots)
        last_shot = rally_shots[-1]

        server_id_for_this_rally = 'Unknown'
        # Try to find the server ID for this point/rally_idx from serves_info_list
        # This relies on 'point_num' being correctly added to serve_info dicts
        for si in reversed(serves_info_list):
            if si.get('point_num') == current_point_num:
                if si.get('is_in') and not si.get('is_fault'):
                    server_id_for_this_rally = si.get('server_id', 'Unknown')
                    break
                # If no "IN" serve found for this point_num, try to get server from last fault for this point_num
                if server_id_for_this_rally == 'Unknown' and si.get('is_fault'):
                     server_id_for_this_rally = si.get('server_id', 'Unknown')
                     # Do not break, prefer an "IN" serve if one exists for this point_num

        if server_id_for_this_rally == 'Unknown' and serves_info_list: # Fallback if point_num not in serves_info or no matching serve
             # This fallback is less reliable as it assumes serves_info_list[rally_idx] is related.
             if rally_idx < len(serves_info_list):
                server_id_for_this_rally = serves_info_list[rally_idx].get('server_id', 'Unknown')


        hitter_of_last_shot = get_player_for_shot(len(rally_shots) - 1, server_id_for_this_rally)
        outcome_details['hitter_of_last_shot'] = hitter_of_last_shot

        last_shot_path = last_shot.get('ball_path')
        if not last_shot_path:
            outcome_details['reason'] = 'Last shot has no ball_path'
            outcome_details['type'] = 'DataError'
            rally_outcomes.append(outcome_details)
            continue

        if detect_net_misses([last_shot_path], net_line_coords) > 0:
            outcome_details['reason'] = f"Net error by {hitter_of_last_shot}"
            outcome_details['type'] = "NetError"
            outcome_details['fault_by_player'] = hitter_of_last_shot
        else:
            last_point_of_last_shot = last_shot_path[-1][:2]
            if detect_out_misses([last_point_of_last_shot], court_boundary_points) > 0:
                outcome_details['reason'] = f"Out error by {hitter_of_last_shot}"
                outcome_details['type'] = "OutError"
                outcome_details['fault_by_player'] = hitter_of_last_shot
            else:
                opponent = "Unknown"
                if hitter_of_last_shot == "player1": opponent = "player2"
                elif hitter_of_last_shot == "player2": opponent = "player1"
                outcome_details['reason'] = f"Effective Winner by {hitter_of_last_shot} (opponent {opponent} failed to return)"
                outcome_details['type'] = "EffectiveWinner"
                outcome_details['winner'] = hitter_of_last_shot

        rally_outcomes.append(outcome_details)
    return rally_outcomes
