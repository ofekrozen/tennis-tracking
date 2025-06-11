import argparse
import queue
import pandas as pd 
import pickle
import imutils
import os
from PIL import Image, ImageDraw
import cv2 
import numpy as np
import torch
import sys
import time

from sktime.datatypes._panel._convert import from_2d_array_to_nested
from court_detector import CourtDetector
from Models.tracknet import trackNet
from TrackPlayers.trackplayers import *
from utils import get_video_properties, get_dtype
from detection import *
from pickle import load

from statistics_extraction.shot_accuracy_stats import detect_out_misses, detect_net_misses, check_serve_accuracy
from statistics_extraction.utils import is_point_in_polygon # Added import
from court_reference import CourtReference # Ensure CourtReference is available
from statistics_extraction.rally_stats import identify_rallies_and_shots, calculate_longest_rally, analyze_rally_outcomes
from statistics_extraction.player_activity_stats import check_player_net_activity


# parse parameters
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")
parser.add_argument("--minimap", type=int, default=0)
parser.add_argument("--bounce", type=int, default=0)

args = parser.parse_args()

input_video_path = args.input_video_path
output_video_path = args.output_video_path
minimap = args.minimap
bounce = args.bounce

n_classes = 256
save_weights_path = 'WeightsTracknet/model.1'
yolo_classes = 'Yolov3/yolov3.txt'
yolo_weights = 'Yolov3/yolov3.weights'
yolo_config = 'Yolov3/yolov3.cfg'

if output_video_path == "":
    # output video in same path
    output_video_path = input_video_path.split('.')[0] + "VideoOutput/video_output.mp4"

# get video fps&video size
video = cv2.VideoCapture(input_video_path)
fps = int(video.get(cv2.CAP_PROP_FPS))
print('fps : {}'.format(fps))
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# try to determine the total number of frames in the video file
if imutils.is_cv2() is True :
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
else : 
    prop = cv2.CAP_PROP_FRAME_COUNT
total = int(video.get(prop))

# start from first frame
currentFrame = 0

# width and height in TrackNet
width, height = 640, 360
img, img1, img2 = None, None, None

# load TrackNet model
modelFN = trackNet
m = modelFN(n_classes, input_height=height, input_width=width)
m.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
m.load_weights(save_weights_path)

# In order to draw the trajectory of tennis, we need to save the coordinate of previous 7 frames
q = queue.deque()
for i in range(0, 8):
    q.appendleft(None)

# save prediction images as videos
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

# load yolov3 labels
LABELS = open(yolo_classes).read().strip().split("\n")
# yolo net
net = cv2.dnn.readNet(yolo_weights, yolo_config)

# court
court_detector = CourtDetector()

# players tracker
dtype = get_dtype()
detection_model = DetectionModel(dtype=dtype)

# get videos properties
fps, length, v_width, v_height = get_video_properties(video)

coords = []
frame_i = 0
frames = []
t = []

while True:
  ret, frame = video.read()
  frame_i += 1

  if ret:
    if frame_i == 1:
      print('Detecting the court and the players...')
      lines = court_detector.detect(frame)
    else: # then track it
      lines = court_detector.track_court(frame)
    detection_model.detect_player_1(frame, court_detector)
    detection_model.detect_top_persons(frame, court_detector, frame_i)
    
    for i_loop in range(0, len(lines), 4): # Renamed loop variable
      x1, y1, x2, y2 = lines[i_loop],lines[i_loop+1], lines[i_loop+2], lines[i_loop+3]
      cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 5)
    new_frame = cv2.resize(frame, (v_width, v_height))
    frames.append(new_frame)
  else:
    break
video.release()
print('Finished!')

# Initialize transformed_center_line here so it's in scope
transformed_center_line = None
transformed_court_boundary_points = [] # Ensure it's initialized
net_line_coords_for_stats = [] # Ensure it's initialized
transformed_service_boxes = {} # Ensure it's initialized

if court_detector.court_warp_matrix is not None and len(court_detector.court_warp_matrix) > 0:
    reference_court_points = np.array([
        [286, 561],  # Top-left
        [1379, 561], # Top-right
        [1379, 2935],# Bottom-right
        [286, 2935]  # Bottom-left
    ], dtype=np.float32).reshape((-1, 1, 2))

    homography_matrix = court_detector.court_warp_matrix[-1]
    temp_transformed_court_boundary_points = cv2.perspectiveTransform(reference_court_points, homography_matrix)
    if temp_transformed_court_boundary_points is not None:
        transformed_court_boundary_points = [tuple(p[0]) for p in temp_transformed_court_boundary_points]
        print(f"DEBUG_VALIDATE: Transformed Court Boundary Points: {transformed_court_boundary_points}")
    else:
        print("Warning: cv2.perspectiveTransform returned None for court boundary points.")
        transformed_court_boundary_points = [] # Ensure it's an empty list

    court_ref_for_center_line = CourtReference()
    if hasattr(court_ref_for_center_line, 'middle_line'):
        ref_center_line_pts = np.array([court_ref_for_center_line.middle_line[0], court_ref_for_center_line.middle_line[1]], dtype=np.float32).reshape((-1, 1, 2))
        transformed_center_line_pts = cv2.perspectiveTransform(ref_center_line_pts, homography_matrix)
        if transformed_center_line_pts is not None and len(transformed_center_line_pts) == 2:
            transformed_center_line = (tuple(transformed_center_line_pts[0][0]), tuple(transformed_center_line_pts[1][0]))
            print(f"DEBUG_VALIDATE: Transformed Center Line: {transformed_center_line}")
        else:
            print("Warning: Could not transform center line.")
            transformed_center_line = None # Ensure it's None
    else:
        print("Warning: CourtReference has no middle_line attribute for center line.")
        transformed_center_line = None # Ensure it's None
else:
    print("Warning: Court warp matrix not available or empty. Cannot calculate out misses accurately.")


if hasattr(court_detector, 'lines') and court_detector.lines is not None and len(court_detector.lines) >= 12:
    net_line_coords_for_stats = court_detector.lines[8:12]
    print(f"DEBUG_VALIDATE: Net Line Coords for Stats: {net_line_coords_for_stats}")
else:
    print("Warning: Court lines (including net) not available or not in expected format. Cannot calculate net misses accurately.")

court_ref = CourtReference()
reference_service_boxes = court_ref.get_service_box_references()
if court_detector.court_warp_matrix is not None and len(court_detector.court_warp_matrix) > 0 and 'homography_matrix' in locals():
    for box_name, raw_points in reference_service_boxes.items():
        np_raw_points = np.array(raw_points, dtype=np.float32).reshape((-1, 1, 2))
        transformed_points = cv2.perspectiveTransform(np_raw_points, homography_matrix)
        if transformed_points is not None:
            transformed_service_boxes[box_name] = [tuple(p[0]) for p in transformed_points]
        else:
            print(f"Warning: cv2.perspectiveTransform returned None for service box {box_name}")
else:
    print("Warning: Court warp matrix not available. Cannot determine service boxes accurately for serve statistics.")

detection_model.find_player_2_box()

ball_landing_positions = []
# ball_trajectories_for_net_miss = []
rallies_data = []
all_detected_bounces_for_rallies = []

# Game state variables for serve detection cycle
point_currently_in_progress = False
rally_ended_in_last_segment = True
player_who_served_last = None
is_first_serve_attempt_for_point = True
# transformed_center_line is initialized and potentially set above

player1_boxes = detection_model.player_1_boxes
player2_boxes = detection_model.player_2_boxes

video = cv2.VideoCapture(input_video_path)
frame_i = 0
currentFrame = 0

last = time.time()
for img_frame_loop in frames:
    output_img = img_frame_loop.copy()

    img_resized = cv2.resize(output_img, (width, height))
    img_float = img_resized.astype(np.float32)
    X = np.rollaxis(img_float, 2, 0)
    pr = m.predict(np.array([X]))[0]
    pr = pr.reshape((height, width, n_classes)).argmax(axis=2)
    pr = pr.astype(np.uint8)
    heatmap = cv2.resize(pr, (output_width, output_height))
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7)

    # Marking player boxes - ensure playerX_boxes are lists of boxes for *each frame* if mark_player_box expects that for currentFrame index
    # If playerX_boxes from detection_model are single boxes (last known), this needs adjustment or mark_player_box needs to handle it.
    # Assuming detection_model.player_1_boxes is a list of boxes per frame from the initial pass
    current_p1_box = detection_model.player_1_boxes[currentFrame] if currentFrame < len(detection_model.player_1_boxes) else None
    current_p2_box = detection_model.player_2_boxes[currentFrame] if currentFrame < len(detection_model.player_2_boxes) else None

    if current_p1_box and current_p1_box[0] is not None: # If a box exists for this frame
        output_img_marked = mark_player_box(output_img, [current_p1_box], 0) # Pass as list with one item
    else:
        output_img_marked = output_img # No box to mark or already marked via detection_model attributes

    if current_p2_box and current_p2_box[0] is not None:
        output_img_marked = mark_player_box(output_img_marked, [current_p2_box], 0)
    
    PIL_image = cv2.cvtColor(output_img_marked, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)

    if circles is not None:
        if len(circles) == 1:
            x, y = int(circles[0][0][0]), int(circles[0][0][1])
            coords.append([x,y]); t.append(time.time()-last); q.appendleft([x, y]); q.pop()
        else:
            coords.append(None); t.append(time.time()-last); q.appendleft(None); q.pop()
    else:
        coords.append(None); t.append(time.time()-last); q.appendleft(None); q.pop()

    for i_q_item in range(0, 8):
        if q[i_q_item] is not None:
            draw_x, draw_y = q[i_q_item][0], q[i_q_item][1]
            bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
            draw = ImageDraw.Draw(PIL_image); draw.ellipse(bbox, outline='yellow'); del draw
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)
    output_video.write(opencvImage)
    currentFrame += 1

video.release(); output_video.release()

if minimap == 1:
  game_video = cv2.VideoCapture(output_video_path)
  output_video_map = cv2.VideoWriter('VideoOutput/video_with_map.mp4', fourcc, fps, (output_width, output_height))
  print('Adding the mini-map...')
  temp_coords_for_minimap = [c for c in coords if c is not None]
  if len(temp_coords_for_minimap) > 1:
    x_m, y_m = diff_xy(temp_coords_for_minimap); remove_outliers(x_m, y_m, temp_coords_for_minimap)
    temp_coords_for_minimap = interpolation(temp_coords_for_minimap)
    # Ensure detection_model still has valid player boxes if create_top_view uses them
    create_top_view(court_detector, detection_model, temp_coords_for_minimap, fps)
    minimap_video = cv2.VideoCapture('VideoOutput/minimap.mp4')
    if minimap_video.isOpened():
        while True:
          ret_gv, frame_gv = game_video.read(); ret_mv, img_mv = minimap_video.read()
          if ret_gv: output_video_map.write(merge(frame_gv, img_mv) if ret_mv else frame_gv)
          else: break
        minimap_video.release()
    else:
        print("Warning: Minimap video file could not be opened.")
    output_video_map.release()
  else: print("Not enough ball coordinates for minimap.")
  game_video.release()

for _ in range(3):
  if len(coords) > 1: x_coords_post, y_coords_post = diff_xy(coords); remove_outliers(x_coords_post, y_coords_post, coords) # Renamed x,y vars
coords = interpolation(coords)

coords_with_dummy_frame = []
if coords:
    coords_with_dummy_frame = [(pt[0], pt[1], i_pt) for i_pt, pt in enumerate(coords) if pt is not None]

if coords_with_dummy_frame:
    player_positions_for_ace = {}
    # num_total_frames_for_ace = total # This 'total' is total frames in video, not necessarily in coords_with_dummy_frame
    # Use length of detection_model.player_X_boxes if available, or fallback to len(coords_with_dummy_frame)
    # detection_model.player_1_boxes etc are populated during the first video read loop
    if hasattr(detection_model, 'player_1_boxes') and detection_model.player_1_boxes:
        player_positions_for_ace['player1'] = []
        for frame_num, box in enumerate(detection_model.player_1_boxes):
            if box[0] is not None:
                xt, yt, xb, yb = box; x_player, y_player, w_player, h_player = xt, yt, xb - xt, yb - yt # Renamed local vars
                player_positions_for_ace['player1'].append((x_player, y_player, w_player, h_player, frame_num))
    if hasattr(detection_model, 'player_2_boxes') and detection_model.player_2_boxes:
        player_positions_for_ace['player2'] = []
        for frame_num, box in enumerate(detection_model.player_2_boxes):
            if box[0] is not None:
                xt, yt, xb, yb = box; x_player, y_player, w_player, h_player = xt, yt, xb - xt, yb - yt # Renamed local vars
                player_positions_for_ace['player2'].append((x_player, y_player, w_player, h_player, frame_num))
    print(f"DEBUG_VALIDATE: Prepared player_positions_for_ace: P1 has {len(player_positions_for_ace.get('player1',[]))} tracked frames, P2 has {len(player_positions_for_ace.get('player2',[]))} tracked frames.")

    if transformed_court_boundary_points and net_line_coords_for_stats : # Ensure these are valid
        rallies_data = identify_rallies_and_shots( ball_trajectory=coords_with_dummy_frame, player_positions=None, court_info=None,
            net_line_coords=net_line_coords_for_stats, court_boundary_points=transformed_court_boundary_points,
            all_detected_bounces=all_detected_bounces_for_rallies )
        print(f"DEBUG: Shot segmentation resulted in {len(rallies_data)} rally/rallies.")
        if rallies_data and len(rallies_data) > 0 and rallies_data[0]: print(f"DEBUG: First rally has {len(rallies_data[0])} shots.")
        if rallies_data: total_shots_in_rallies = sum(len(rally) for rally in rallies_data); print(f"DEBUG_VALIDATE: identify_rallies_and_shots call resulted in {len(rallies_data)} rallies with a total of {total_shots_in_rallies} shots.")

    serves_info_list = []
    print(f"DEBUG_VALIDATE: Starting Serve Detection Cycle based on {len(rallies_data)} detected rallies.")
    point_num = 0
    num_potential_points = len(rallies_data) + 1
    while point_num < num_potential_points:
        current_point_processed_and_ended = False
        print(f"DEBUG_VALIDATE: Processing Point #{point_num + 1}. is_first_serve_attempt: {is_first_serve_attempt_for_point}, rally_ended_prior: {rally_ended_in_last_segment}")

        if not rally_ended_in_last_segment and point_num > 0 :
            if (point_num - 1) < len(rallies_data):
                 print(f"DEBUG_VALIDATE: Rally for previous point ({point_num}) existed. Setting rally_ended_in_last_segment=True.")
                 rally_ended_in_last_segment = True
            else:
                 print(f"DEBUG_VALIDATE: Serve was IN for point ({point_num}) but no rally data followed. Ending point.")
                 rally_ended_in_last_segment = True
            if not rally_ended_in_last_segment:
                print(f"DEBUG_VALIDATE: Point {point_num + 1} serve opportunity skipped, previous rally outcome still pending.")
                point_num +=1 ; continue

        if not is_first_serve_attempt_for_point and not rally_ended_in_last_segment:
            print(f"DEBUG_VALIDATE: State error - trying 2nd serve for point {point_num+1}, but rally was already in progress. Advancing point.")
            current_point_processed_and_ended = True; is_first_serve_attempt_for_point = True; rally_ended_in_last_segment = True; point_num +=1; continue

        serve_analysis_trajectory_start_frame = 0
        first_serve_attempt_trajectory_start_frame_for_point = -1

        current_serve_ball_trajectory = [] # Initialize to ensure it's always defined
        if is_first_serve_attempt_for_point:
            if point_num > 0 and point_num <= len(rallies_data) and rally_ended_in_last_segment:
                if rallies_data[point_num-1]: serve_analysis_trajectory_start_frame = rallies_data[point_num-1][-1]['end_frame'] + 1
            elif point_num == 0 : serve_analysis_trajectory_start_frame = 0
            current_serve_ball_trajectory = [p for p in coords_with_dummy_frame if p[2] >= serve_analysis_trajectory_start_frame]
            if current_serve_ball_trajectory: # Store start frame of actual motion for 1st serve
                 first_serve_attempt_trajectory_start_frame_for_point = current_serve_ball_trajectory[0][2]

        else: # 2nd serve
            if len(serves_info_list) > 0:
                last_serve_event = serves_info_list[-1]
                if last_serve_event.get('server_id') != player_who_served_last or not last_serve_event.get('is_first_serve') or not last_serve_event.get('is_fault'):
                    print(f"DEBUG_VALIDATE: State error for 2nd serve. Last serve by {last_serve_event.get('server_id')}, current server {player_who_served_last}. Or not 1st serve fault.")
                    current_point_processed_and_ended = True
                else:
                    if last_serve_event['frame_of_bounce'] > 0:
                        serve_analysis_trajectory_start_frame = last_serve_event['frame_of_bounce'] + fps // 3
                    elif first_serve_attempt_trajectory_start_frame_for_point > 0 :
                         serve_analysis_trajectory_start_frame = first_serve_attempt_trajectory_start_frame_for_point + fps
                    else: # Fallback if everything else fails
                         serve_analysis_trajectory_start_frame = (coords_with_dummy_frame[0][2] if coords_with_dummy_frame else 0) + fps
                current_serve_ball_trajectory = [p for p in coords_with_dummy_frame if p[2] >= serve_analysis_trajectory_start_frame]


        if not current_serve_ball_trajectory or len(current_serve_ball_trajectory) < 3:
            serve_attempt_type_str = '1st' if is_first_serve_attempt_for_point else '2nd'
            print(f"DEBUG_VALIDATE: Point {point_num+1} ({serve_attempt_type_str} Serve): Not enough ball trajectory data after frame {serve_analysis_trajectory_start_frame}.")
            if not is_first_serve_attempt_for_point:
                 serves_info_list.append({ 'landing_coord': None, 'is_first_serve': False,
                                           'target_service_box_transformed_points': actual_target_box_transformed if 'actual_target_box_transformed' in locals() and actual_target_box_transformed else "N/A_NO_TRAJ_2ND",
                                           'frame_of_bounce': -1, 'server_id': player_who_served_last,
                                           'served_from_deuce': serving_from_deuce_side if 'serving_from_deuce_side' in locals() else "N/A",
                                           'is_in': False, 'is_fault': True, 'is_double_fault': True, 'point_num': point_num })
            current_point_processed_and_ended = True

        if current_point_processed_and_ended: point_num +=1; is_first_serve_attempt_for_point = True; rally_ended_in_last_segment = True; continue

        initial_ball_pos_for_serve = current_serve_ball_trajectory[0][:2]

        server_id = None; server_half = None; serving_from_deuce_side = False; target_box_name = None; actual_target_box_transformed = None

        if is_first_serve_attempt_for_point:
            # --- Start of Full Server/Side/Target Logic for 1st Serve of a Point ---
            print(f"DEBUG_VALIDATE: Point {point_num+1} (1st Serve Attempt): Determining server and target.")
            if not transformed_court_boundary_points:
                print(f"DEBUG_VALIDATE: Point {point_num+1} (1st Serve): CRITICAL - Missing court boundaries for server ID. Skipping point.");
                current_point_processed_and_ended = True
            else:
                baseline_y_top = (transformed_court_boundary_points[0][1] + transformed_court_boundary_points[1][1]) / 2
                baseline_y_bottom = (transformed_court_boundary_points[2][1] + transformed_court_boundary_points[3][1]) / 2
                court_center_y = (baseline_y_top + baseline_y_bottom) / 2
                prospective_server_id_based_on_alternation = None
                if point_num > 0 or (point_num == 0 and player_who_served_last is not None):
                    if player_who_served_last == "player1": prospective_server_id_based_on_alternation = "player2"
                    elif player_who_served_last == "player2": prospective_server_id_based_on_alternation = "player1"

                p1_box_defined = detection_model.player_1_box is not None
                p2_box_defined = detection_model.player_2_box is not None

                if prospective_server_id_based_on_alternation == "player1" and p1_box_defined:
                    p1_box_center_y = detection_model.player_1_box[1] + detection_model.player_1_box[3] / 2
                    if abs(p1_box_center_y - baseline_y_bottom) < output_height * 0.20 and initial_ball_pos_for_serve[1] > court_center_y:
                        server_id = "player1"; server_half = "bottom"
                elif prospective_server_id_based_on_alternation == "player2" and p2_box_defined:
                    p2_box_center_y = detection_model.player_2_box[1] + detection_model.player_2_box[3] / 2
                    if abs(p2_box_center_y - baseline_y_top) < output_height * 0.20 and initial_ball_pos_for_serve[1] < court_center_y:
                        server_id = "player2"; server_half = "top"

                if not server_id:
                    print(f"DEBUG_VALIDATE: Server not set by alternation (prospective: {prospective_server_id_based_on_alternation}), attempting full positional detection for Point {point_num+1}.")
                    if p1_box_defined:
                        p1_box_center_y = detection_model.player_1_box[1] + detection_model.player_1_box[3] / 2
                        if abs(p1_box_center_y - baseline_y_bottom) < output_height * 0.20 and initial_ball_pos_for_serve[1] > court_center_y:
                            server_id = "player1"; server_half = "bottom"
                    if not server_id and p2_box_defined:
                        p2_box_center_y = detection_model.player_2_box[1] + detection_model.player_2_box[3] / 2
                        if abs(p2_box_center_y - baseline_y_top) < output_height * 0.20 and initial_ball_pos_for_serve[1] < court_center_y:
                            server_id = "player2"; server_half = "top"

                if not server_id:
                    print(f"DEBUG_VALIDATE: Point {point_num+1} (1st Serve): CRITICAL - Could not identify server. Skipping point's serve processing.")
                    current_point_processed_and_ended = True
                else:
                    player_who_served_last = server_id
                    print(f"DEBUG_VALIDATE: Point {point_num+1} (1st Serve): Confirmed Server ID: {server_id} ({server_half} half). Initial ball: {initial_ball_pos_for_serve}")
                    if transformed_center_line:
                        center_line_avg_x = (transformed_center_line[0][0] + transformed_center_line[1][0]) / 2
                        server_pos_x = initial_ball_pos_for_serve[0]
                        if server_id == "player1" and p1_box_defined: server_pos_x = detection_model.player_1_box[0] + detection_model.player_1_box[2] / 2
                        elif server_id == "player2" and p2_box_defined: server_pos_x = detection_model.player_2_box[0] + detection_model.player_2_box[2] / 2
                        if server_half == "bottom": serving_from_deuce_side = server_pos_x > center_line_avg_x
                        elif server_half == "top": serving_from_deuce_side = server_pos_x < center_line_avg_x
                        print(f"DEBUG_VALIDATE: Point {point_num+1} (1st Serve): Server {server_id} serving from Deuce: {serving_from_deuce_side} (ServerX: {server_pos_x:.0f}, CenterLineX: {center_line_avg_x:.0f})")
                    else:
                        print(f"DEBUG_VALIDATE: Point {point_num+1} (1st Serve): Cannot determine Deuce/Ad side (transformed_center_line missing). Defaulting to Deuce.")
                        serving_from_deuce_side = True

                    if server_half == "bottom": target_box_name = "top_left_box" if serving_from_deuce_side else "top_right_box"
                    elif server_half == "top": target_box_name = "bottom_left_box" if serving_from_deuce_side else "bottom_right_box"
                    else: print(f"DEBUG_VALIDATE: Point {point_num+1} (1st Serve): CRITICAL - server_half not set for server {server_id}. Skipping point."); current_point_processed_and_ended = True

                    if not current_point_processed_and_ended : actual_target_box_transformed = transformed_service_boxes.get(target_box_name)
                    if not actual_target_box_transformed and not current_point_processed_and_ended:
                        print(f"DEBUG_VALIDATE: Point {point_num+1} (1st Serve): CRITICAL - Could not get target box '{target_box_name}'. Skipping point.")
                        current_point_processed_and_ended = True
            # --- End of Full Server/Side/Target Logic ---
            else: # For 2nd serve
                if serves_info_list:
                    last_serve_info = serves_info_list[-1]
                    if last_serve_info.get('server_id') == player_who_served_last and last_serve_info.get('is_first_serve'):
                        server_id = last_serve_info['server_id']
                        server_half = "bottom" if server_id == "player1" else "top"
                        serving_from_deuce_side = last_serve_info['served_from_deuce']
                        actual_target_box_transformed = last_serve_info['target_service_box_transformed_points']
                        if server_half == "bottom": target_box_name = "top_left_box" if serving_from_deuce_side else "top_right_box"
                        else: target_box_name = "bottom_left_box" if serving_from_deuce_side else "bottom_right_box"
                        print(f"DEBUG_VALIDATE: Point {point_num+1} (2nd Serve): Server {server_id}, Side Deuce {serving_from_deuce_side}, Target {target_box_name}")
                    else:
                        print(f"DEBUG_VALIDATE: Point {point_num+1} (2nd Serve): State error - last serve info mismatch. Skipping 2nd serve.")
                        current_point_processed_and_ended = True
                else:
                    print(f"DEBUG_VALIDATE: Point {point_num+1} (2nd Serve): State error - no 1st serve recorded. Skipping 2nd serve.")
                    current_point_processed_and_ended = True

            if current_point_processed_and_ended:
                point_num +=1; is_first_serve_attempt_for_point = True; rally_ended_in_last_segment = True; continue

            if not actual_target_box_transformed :
                print(f"DEBUG_VALIDATE: Point {point_num+1}: Target box points missing for {'1st' if is_first_serve_attempt_for_point else '2nd'} serve. Skipping point.")
                point_num +=1; is_first_serve_attempt_for_point = True; rally_ended_in_last_segment = True; continue

            serve_landing_coord = None; serve_bounce_frame = -1; serve_is_in = False; serve_is_fault = True; is_net_on_serve = False

            serve_path_for_net_check = current_serve_ball_trajectory[:20]
            if net_line_coords_for_stats and detect_net_misses([serve_path_for_net_check], net_line_coords_for_stats) > 0:
                is_net_on_serve = True; print(f"DEBUG_VALIDATE: Point {point_num+1} ({'1st' if is_first_serve_attempt_for_point else '2nd'} Serve): Detected NET on trajectory by {server_id}.")

            if bounce == 1 and all_detected_bounces_for_rallies and coords_with_dummy_frame:
                max_serve_landing_frame = current_serve_ball_trajectory[0][2] + (fps * 3) if current_serve_ball_trajectory else 0
                relevant_bounces = [b for b in all_detected_bounces_for_rallies if b[2] >= current_serve_ball_trajectory[0][2] and b[2] <= max_serve_landing_frame] if current_serve_ball_trajectory else []
                if relevant_bounces:
                    relevant_bounces.sort(key=lambda b: b[2])
                    serve_landing_coord = (relevant_bounces[0][0], relevant_bounces[0][1])
                    serve_bounce_frame = relevant_bounces[0][2]
                    if actual_target_box_transformed : serve_is_in = is_point_in_polygon(serve_landing_coord, actual_target_box_transformed)

            if is_net_on_serve and serve_is_in:
                print(f"DEBUG_VALIDATE: Point {point_num+1} ({'1st' if is_first_serve_attempt_for_point else '2nd'} Serve): LET by {server_id}. Replaying serve.")
                rally_ended_in_last_segment = True; current_point_processed_and_ended = False
            elif not is_net_on_serve and serve_is_in: serve_is_fault = False
            else: serve_is_fault = True

            serves_info_list.append({'landing_coord': serve_landing_coord, 'is_first_serve': is_first_serve_attempt_for_point,
                'target_service_box_transformed_points': actual_target_box_transformed, 'frame_of_bounce': serve_bounce_frame, 'server_id': server_id,
                'served_from_deuce': serving_from_deuce_side, 'is_in': serve_is_in, 'is_fault': serve_is_fault,
                'is_double_fault': (not is_first_serve_attempt_for_point and serve_is_fault), 'point_num': point_num
            })
            print(f"DEBUG_VALIDATE: Point {point_num+1} ({'1st' if is_first_serve_attempt_for_point else '2nd'} Serve) by {server_id}: TargetBox={target_box_name if target_box_name else 'N/A'}, Bounce@F{serve_bounce_frame}, Coord={serve_landing_coord}, In={serve_is_in}, Fault={serve_is_fault}, NetHit={is_net_on_serve}")

            if not (is_net_on_serve and serve_is_in): # If not a LET that needs replaying this attempt
                if is_first_serve_attempt_for_point:
                    if serve_is_fault: # First serve fault
                        is_first_serve_attempt_for_point = False
                        rally_ended_in_last_segment = True # Prepare for 2nd serve, point not over
                        current_point_processed_and_ended = False
                    else: # First serve is IN
                        point_currently_in_progress = True # Rally starts
                        rally_ended_in_last_segment = False # Rally will determine when point ends
                        current_point_processed_and_ended = False # Waiting for rally to conclude
                else: # This was a 2nd serve attempt
                    current_point_processed_and_ended = True # Point ends after 2nd serve (fault or in)
                    is_first_serve_attempt_for_point = True # Reset for next point
                    rally_ended_in_last_segment = True # Point ended

            if current_point_processed_and_ended:
                point_num += 1
                if (point_num-1) < len(rallies_data): rally_ended_in_last_segment = True
                else: rally_ended_in_last_segment = True # If no rally followed (e.g. ace) or end of rallies

# velocty 
Vx = []
Vy = []
V = []
frames = [*range(len(coords))]

for i in range(len(coords)-1):
  p1 = coords[i]
  p2 = coords[i+1]
  t1 = t[i]
  t2 = t[i+1]
  x = (p1[0]-p2[0])/(t1-t2)
  y = (p1[1]-p2[1])/(t1-t2)
  Vx.append(x)
  Vy.append(y)

for i in range(len(Vx)):
  vx = Vx[i]
  vy = Vy[i]
  v = (vx**2+vy**2)**0.5
  V.append(v)

xy = coords[:]

if bounce == 1:
  # Predicting Bounces 
  test_df = pd.DataFrame({'x': [coord[0] for coord in xy[:-1]], 'y':[coord[1] for coord in xy[:-1]], 'V': V})

  # df.shift
  for i in range(20, 0, -1): 
    test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
  for i in range(20, 0, -1): 
    test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
  for i in range(20, 0, -1): 
    test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)

  test_df.drop(['x', 'y', 'V'], 1, inplace=True)

  Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
        'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
        'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
        'lagX_2', 'lagX_1']]
  Xs = from_2d_array_to_nested(Xs.to_numpy())

  Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
        'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
        'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
        'lagY_3', 'lagY_2', 'lagY_1']]
  Ys = from_2d_array_to_nested(Ys.to_numpy())

  Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
        'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
        'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
        'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
  Vs = from_2d_array_to_nested(Vs.to_numpy())

  X = pd.concat([Xs, Ys, Vs], 1)

  # load the pre-trained classifier  
  clf = load(open('clf.pkl', 'rb'))

  predcted = clf.predict(X)
  idx = list(np.where(predcted == 1)[0])
  idx = np.array(idx) - 10
  
  # Collect ball landing positions for out miss detection
  if transformed_court_boundary_points: # Only collect if court boundaries are available
      for i_bounce_idx in idx: # Renamed loop variable for clarity
          if 0 <= i_bounce_idx < len(xy): # Ensure index is valid
              # xy elements are [x,y] coordinates from interpolated 'coords'
              current_landing_pos = tuple(xy[i_bounce_idx])
              ball_landing_positions.append(current_landing_pos)
              print(f"DEBUG_VALIDATE: Collected for 'Total Out Misses' summary: Landing Coords={current_landing_pos}, Original Idx in xy={i_bounce_idx}")

              # Populate all_detected_bounces_for_rallies
              # This assumes `coords_with_dummy_frame` is available and aligned with `xy` regarding bounce indices.
              # The `idx` are indices for `xy`. `xy` is `coords[:]` after interpolation.
              # `coords_with_dummy_frame` adds frame numbers: `[(pt[0], pt[1], i) for i, pt in enumerate(coords) if pt is not None]`
              # This alignment can be tricky. A robust solution would map `idx` (indices of `xy`)
              # back to original frame numbers more directly if `coords_with_dummy_frame` was the source for `xy`.
              # Given current structure, this is an approximation.
              if 'coords_with_dummy_frame' in locals() and coords_with_dummy_frame:
                  if 0 <= i_bounce_idx < len(coords_with_dummy_frame): # Check index for coords_with_dummy_frame
                      bounce_x, bounce_y = xy[i_bounce_idx]
                      frame_num = coords_with_dummy_frame[i_bounce_idx][2] # Get frame_num
                      all_detected_bounces_for_rallies.append((bounce_x, bounce_y, frame_num))
                  # else:
                      # Fallback or warning if idx is out of bounds for coords_with_dummy_frame after being valid for xy
                      # print(f"DEBUG: Bounce index {i_bounce_idx} valid for xy but out of range for coords_with_dummy_frame (len {len(coords_with_dummy_frame)})")
            # else:
                # print("DEBUG: coords_with_dummy_frame not available for populating all_detected_bounces_for_rallies")

  # print(f"DEBUG: Populated {len(all_detected_bounces_for_rallies)} bounces for rally analysis.")

  if minimap == 1:
    video = cv2.VideoCapture('VideoOutput/video_with_map.mp4')
  else:
    video = cv2.VideoCapture(output_video_path)

  output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = int(video.get(cv2.CAP_PROP_FPS))
  length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  fourcc = cv2.VideoWriter_fourcc(*'XVID')

  print(fps)
  print(length)

  output_video = cv2.VideoWriter('VideoOutput/final_video.mp4', fourcc, fps, (output_width, output_height))
  i_loop_bounce_viz = 0 # Renamed loop variable
  while True:
    ret, frame = video.read()
    if ret:
      # if coords[i] is not None:
      if i_loop_bounce_viz in idx: # Use renamed loop variable
        center_coordinates = int(xy[i_loop_bounce_viz][0]), int(xy[i_loop_bounce_viz][1]) # Use renamed loop variable
        radius = 3
        color = (255, 0, 0)
        thickness = -1
        cv2.circle(frame, center_coordinates, 10, color, thickness)
      i_loop_bounce_viz += 1 # Use renamed loop variable
      output_video.write(frame)
    else:
      break

  video.release()
  output_video.release()

if rallies_data and net_line_coords_for_stats:
    all_shots_for_net_detection = []
    rally_idx_debug = 0 # Initialize rally index for debug
    for rally in rallies_data:
        for shot_idx_debug, shot in enumerate(rally): # Added shot_idx_debug for logging
            all_shots_for_net_detection.append(shot['ball_path'])
            if shot['ball_path']: # Ensure path is not empty
                print(f"DEBUG_VALIDATE: Added shot for Net Miss Check: RallyIndex={rally_idx_debug}, ShotIndexInRally={shot_idx_debug}, Frames {shot['ball_path'][0][2]}-{shot['ball_path'][-1][2]}, Points={len(shot['ball_path'])}")
        rally_idx_debug += 1 # Increment rally index for debug

    if all_shots_for_net_detection:
        total_net_misses = detect_net_misses(all_shots_for_net_detection, net_line_coords_for_stats)
        print(f"Total net misses (from segmented shots): {total_net_misses}")
    else:
        print("No shots detected to check for net misses.")
elif not net_line_coords_for_stats:
    print("Could not calculate net misses: Net line coordinates not determined.")
else:
    print("Could not calculate net misses: No rally/shot data available.")

if serves_info_list: # If any serves were detected and processed
    # Call Ace Detection
    total_aces = detect_aces(serves_info_list, coords_with_dummy_frame, player_positions_for_ace, fps)
    print(f"Total Aces: {total_aces}")

    first_in, first_out, second_in, second_out = check_serve_accuracy(serves_info_list)
    print(f"Serve Accuracy: 1st Serves In: {first_in}, 1st Serves Out/Fault: {first_out}")
    print(f"                2nd Serves In: {second_in}, 2nd Serves Out/Fault (Double Faults): {second_out}")
    if (first_in + first_out) > 0:
        first_serve_percentage = (first_in / (first_in + first_out)) * 100
        print(f"                1st Serve Percentage: {first_serve_percentage:.2f}%")
elif not transformed_service_boxes:
    print("Could not calculate serve accuracy: Service box coordinates not determined.")
else:
    print("No serve information collected to calculate accuracy.")

if bounce == 1:
    if ball_landing_positions and transformed_court_boundary_points:
        total_out_misses = detect_out_misses(ball_landing_positions, transformed_court_boundary_points)
        print(f"Total out misses (ball landing out of court): {total_out_misses}")
    elif not transformed_court_boundary_points:
        print("Could not calculate out misses: Court boundary points not determined.")
    else:
        print("Could not calculate out misses: No bounce data collected.")
else:
    print("Bounce detection was not enabled (--bounce=0), skipping out miss calculation.")

if rallies_data:
    longest_rally_shots = calculate_longest_rally(rallies_data)
    print(f"Longest rally (by number of shots): {longest_rally_shots} shots")
else:
    print("No rally data to calculate longest rally.")

# --- Points at Net Analysis ---
# Ensure all required variables (rallies_data, player_positions_for_ace, net_line_coords_for_stats, rally_analysis_results)
# are defined and populated before this block.

required_vars_for_net_points = ['rallies_data', 'player_positions_for_ace',
                                'net_line_coords_for_stats', 'rally_analysis_results']
if all(var in locals() and locals()[var] for var in required_vars_for_net_points):

    print("\n--- Points at Net Analysis ---") # Moved print here

    player_net_activity_at_rally_ends = check_player_net_activity(
        rallies_data,
        player_positions_for_ace,
        net_line_coords_for_stats
        # player_ids argument defaults to ['player1', 'player2'] in the function
    )

    points_at_net_won = {'player1': 0, 'player2': 0}
    points_at_net_lost = {'player1': 0, 'player2': 0}
    total_points_played_at_net_by_player = {'player1': 0, 'player2': 0}

    if player_net_activity_at_rally_ends and \
       len(player_net_activity_at_rally_ends) == len(rally_analysis_results):

        for i, net_activity_info in enumerate(player_net_activity_at_rally_ends):
            rally_idx_check = net_activity_info.get('rally_idx') # From check_player_net_activity
            # Find corresponding rally_outcome. Ensure rally_idx matches.
            # Assuming rally_analysis_results is also a list of dicts with 'rally_idx'.
            outcome = None
            for res in rally_analysis_results:
                if res.get('rally_idx') == rally_idx_check:
                    outcome = res
                    break

            if outcome is None:
                print(f"DEBUG_VALIDATE: PointsAtNet - Could not find matching rally outcome for net activity at rally_idx {rally_idx_check}")
                continue

            point_winner = None
            point_loser = None

            # Determine winner/loser from rally outcome
            if outcome.get('type') == "EffectiveWinner":
                point_winner = outcome.get('winner')
                if point_winner == 'player1': point_loser = 'player2'
                elif point_winner == 'player2': point_loser = 'player1'
            elif outcome.get('type') == "NetError" or outcome.get('type') == "OutError":
                point_loser = outcome.get('fault_by_player')
                if point_loser == 'player1': point_winner = 'player2'
                elif point_loser == 'player2': point_winner = 'player1'

            # Check player1 net activity for this rally end
            if net_activity_info.get('player1_at_net'):
                total_points_played_at_net_by_player['player1'] += 1
                if point_winner == 'player1':
                    points_at_net_won['player1'] += 1
                elif point_loser == 'player1': # Or point_winner == 'player2' (if point_winner is not None)
                    points_at_net_lost['player1'] += 1

            # Check player2 net activity for this rally end
            if net_activity_info.get('player2_at_net'):
                total_points_played_at_net_by_player['player2'] += 1
                if point_winner == 'player2':
                    points_at_net_won['player2'] += 1
                elif point_loser == 'player2': # Or point_winner == 'player1' (if point_winner is not None)
                    points_at_net_lost['player2'] += 1

        print(f"Player 1: Points Won at Net: {points_at_net_won['player1']}, Points Lost at Net: {points_at_net_lost['player1']} (out of {total_points_played_at_net_by_player['player1']} points ending with P1 at net)")
        print(f"Player 2: Points Won at Net: {points_at_net_won['player2']}, Points Lost at Net: {points_at_net_lost['player2']} (out of {total_points_played_at_net_by_player['player2']} points ending with P2 at net)")

    else:
        print("Could not perform points at net analysis; data missing or mismatched (rallies, player net activity, or rally outcomes length mismatch).")
else:
    print("Skipping Points at Net analysis: Key data (rallies_data, player_positions_for_ace, net_line_coords_for_stats, or rally_analysis_results) not available or empty.")

if rallies_data:
    rally_analysis_results = analyze_rally_outcomes(rallies_data,
                                                    serves_info_list,
                                                    transformed_court_boundary_points,
                                                    net_line_coords_for_stats)
    print("\n--- Rally Outcome Analysis ---")
    if rally_analysis_results:
        for i, outcome in enumerate(rally_analysis_results):
            print(f"Rally {i+1} ({outcome['num_shots']} shots): Type: {outcome['type']}, Reason: {outcome['reason']}")
    else:
        print("No rally outcome analysis performed.")

[end of predict_video.py]
