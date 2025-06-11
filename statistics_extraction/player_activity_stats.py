import numpy as np

# Define a threshold for being "at the net" (e.g., in pixels from the net line)
# This might need to be dynamic based on court perspective or calibrated.
PLAYER_NET_PROXIMITY_THRESHOLD = 50 # Example value

def check_player_net_activity(player_positions_at_rally_end, net_line_coords, rallies_info):
    """
    Checks if players were near the net at the conclusion of rallies.

    Args:
        player_positions_at_rally_end (dict):
            Maps player_id to their (x, y) position at the end of each rally.
            Example: {
                         'player1': [(x1_r1, y1_r1), (x1_r2, y1_r2), ...],
                         'player2': [(x2_r1, y2_r1), (x2_r2, y2_r2), ...]
                     }
            The index in the list corresponds to the rally index.
        net_line_coords (tuple):
            (x1, y1, x2, y2) representing the net line in image coordinates.
        rallies_info (list of dicts):
            Information about each rally, primarily to know how many rallies there were.
            Could also contain rally end frames to cross-reference player positions if needed.

    Returns:
        list of dicts: For each rally, indicates if players were at the net.
            Example: [
                {'rally_index': 0, 'player1_at_net': False, 'player2_at_net': True},
                {'rally_index': 1, 'player1_at_net': True, 'player2_at_net': False},
                ...
            ]
    """
    net_activity_results = []
    if not net_line_coords or len(net_line_coords) < 4:
        print("Warning: Net line coordinates are invalid for player activity check.")
        return net_activity_results

    net_x1, net_y1, net_x2, net_y2 = net_line_coords
    # For simplicity, we can represent the net line as a segment
    # and calculate player distance to this segment.
    # A more robust way might use the perpendicular distance from player to the infinite line of the net,
    # and check if the player is between the net posts (if posts are detected).

    num_rallies = len(rallies_info)

    for i in range(num_rallies):
        rally_result = {'rally_index': i, 'player1_at_net': False, 'player2_at_net': False} # Default for player IDs

        for player_id, positions_list in player_positions_at_rally_end.items():
            if i < len(positions_list):
                player_pos = positions_list[i]
                px, py = player_pos

                # Simplified distance check: player's y-coordinate relative to average y of net.
                # This assumes a mostly horizontal net on screen.
                # A proper distance to line segment calculation would be better.
                avg_net_y = (net_y1 + net_y2) / 2

                # Check if player is horizontally within the net's span
                is_within_net_x_span = (px >= min(net_x1, net_x2)) and (px <= max(net_x1, net_x2))

                if is_within_net_x_span and abs(py - avg_net_y) < PLAYER_NET_PROXIMITY_THRESHOLD:
                    if player_id == 'player1': # Or however players are identified
                        rally_result['player1_at_net'] = True
                    elif player_id == 'player2':
                        rally_result['player2_at_net'] = True
                    # Extend for more players if necessary

        net_activity_results.append(rally_result)

    return net_activity_results

# Future enhancements:
# - Calculate actual distance from point to line segment for better accuracy.
# - If win/loss data becomes available per rally, this module can be extended
#   to count "points won/lost at the net."
