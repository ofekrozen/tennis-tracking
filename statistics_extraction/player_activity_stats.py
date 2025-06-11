# In statistics_extraction/player_activity_stats.py
import numpy as np

PLAYER_NET_PROXIMITY_THRESHOLD = 75 # Pixels from net line to be considered "at the net"
                                    # Increased from 50 for more tolerance

def check_player_net_activity(rallies_data, player_positions_history, net_line_coords, player_ids=['player1', 'player2']):
    """
    Checks if players were near the net at the conclusion of each rally.

    Args:
        rallies_data (list of list of dicts):
            Output from identify_rallies_and_shots. Each rally is a list of shots.
            We need the end_frame of the last shot of each rally.
        player_positions_history (dict):
            Maps player_id to list of (x,y,w,h,frame_num) tuples.
            Example: {'player1': [(x,y,w,h,f), ...], 'player2': [(x,y,w,h,f), ...]}
        net_line_coords (tuple):
            (x1, y1, x2, y2) representing the net line in image coordinates.
        player_ids (list of str): List of player identifiers to check.

    Returns:
        list of dicts: For each rally, indicates which players were at the net.
            Example: [ # Index corresponds to rally_idx
                {'rally_idx': 0, 'player1_at_net': False, 'player2_at_net': True},
                ...
            ]
    """
    net_activity_results = []
    if not net_line_coords or len(net_line_coords) < 4:
        print("DEBUG_VALIDATE: check_player_net_activity: Net line coordinates are invalid.")
        # Return a default structure if net_line_coords is invalid
        for i in range(len(rallies_data)):
            rally_result = {'rally_idx': i}
            for p_id in player_ids:
                rally_result[p_id + '_at_net'] = False
            net_activity_results.append(rally_result)
        return net_activity_results

    net_x1, net_y1, net_x2, net_y2 = net_line_coords
    avg_net_y = (net_y1 + net_y2) / 2
    min_net_x = min(net_x1, net_x2)
    max_net_x = max(net_x1, net_x2)

    for rally_idx, rally_shots in enumerate(rallies_data):
        rally_result = {'rally_idx': rally_idx}
        for p_id in player_ids: rally_result[p_id + '_at_net'] = False # Initialize

        if not rally_shots: # Empty rally
            net_activity_results.append(rally_result)
            continue

        # Ensure 'end_frame' is present in the last shot
        if not rally_shots[-1] or 'end_frame' not in rally_shots[-1]:
            print(f"DEBUG_VALIDATE: Rally {rally_idx} last shot missing 'end_frame'. Skipping net activity check for this rally.")
            net_activity_results.append(rally_result) # Append initialized result
            continue

        rally_end_frame = rally_shots[-1]['end_frame']

        for player_id in player_ids:
            player_history = player_positions_history.get(player_id, [])
            if not player_history: # No history for this player
                rally_result[player_id + '_at_net'] = False
                continue

            player_box_at_rally_end = None
            min_frame_diff = float('inf')

            for x, y, w, h, frame_num in player_history:
                if abs(frame_num - rally_end_frame) < min_frame_diff:
                    min_frame_diff = abs(frame_num - rally_end_frame)
                    player_box_at_rally_end = (x,y,w,h)
                if frame_num > rally_end_frame and min_frame_diff < 5:
                    break

            if player_box_at_rally_end and min_frame_diff < 5:
                px, py, pw, ph = player_box_at_rally_end
                player_feet_y = py + ph
                player_center_x = px + pw / 2

                if min_net_x <= player_center_x <= max_net_x and \
                   abs(player_feet_y - avg_net_y) < PLAYER_NET_PROXIMITY_THRESHOLD:
                    rally_result[player_id + '_at_net'] = True
                    print(f"DEBUG_VALIDATE: Rally {rally_idx}, Player {player_id} AT NET. FeetY: {player_feet_y:.0f}, AvgNetY: {avg_net_y:.0f}, PlayerCenterX: {player_center_x:.0f}, NetSpan: ({min_net_x:.0f}-{max_net_x:.0f})")

        net_activity_results.append(rally_result)

    return net_activity_results
