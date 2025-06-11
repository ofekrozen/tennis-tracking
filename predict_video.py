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
from court_reference import CourtReference # Ensure CourtReference is available
from statistics_extraction.rally_stats import identify_rallies_and_shots, calculate_longest_rally


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
    
    for i in range(0, len(lines), 4):
      x1, y1, x2, y2 = lines[i],lines[i+1], lines[i+2], lines[i+3]
      cv2.line(frame, (int(x1),int(y1)),(int(x2),int(y2)), (0,0,255), 5)
    new_frame = cv2.resize(frame, (v_width, v_height))
    frames.append(new_frame)
  else:
    break
video.release()
print('Finished!')

if court_detector.court_warp_matrix is not None and len(court_detector.court_warp_matrix) > 0:
    # Get reference court boundary points
    # These are for the singles court: (top-left, top-right, bottom-right, bottom-left)
    # from court_reference.py
    # self.left_court_line = ((286, 561), (286, 2935))
    # self.right_court_line = ((1379, 561), (1379, 2935))
    reference_court_points = np.array([
        [286, 561],  # Top-left
        [1379, 561], # Top-right
        [1379, 2935],# Bottom-right
        [286, 2935]  # Bottom-left
    ], dtype=np.float32).reshape((-1, 1, 2)) # Reshape for perspectiveTransform

    homography_matrix = court_detector.court_warp_matrix[-1] # Use the latest homography
    transformed_court_boundary_points = cv2.perspectiveTransform(reference_court_points, homography_matrix)
    transformed_court_boundary_points = [tuple(p[0]) for p in transformed_court_boundary_points] # Convert to list of tuples
else:
    print("Warning: Court warp matrix not available or empty. Cannot calculate out misses accurately.")

if hasattr(court_detector, 'lines') and court_detector.lines is not None and len(court_detector.lines) >= 12:
    # The net line is typically elements 8, 9, 10, 11 (x1, y1, x2, y2)
    # from the 'lines' array returned by court_detector
    net_line_coords_for_stats = court_detector.lines[8:12]
else:
    print("Warning: Court lines (including net) not available or not in expected format. Cannot calculate net misses accurately.")

# Get reference service boxes and transform them
court_ref = CourtReference() # Create an instance to access its methods
reference_service_boxes = court_ref.get_service_box_references()

if court_detector.court_warp_matrix is not None and len(court_detector.court_warp_matrix) > 0: # Check same as for boundary points
    homography_matrix = court_detector.court_warp_matrix[-1]
    for box_name, raw_points in reference_service_boxes.items():
        np_raw_points = np.array(raw_points, dtype=np.float32).reshape((-1, 1, 2))
        transformed_points = cv2.perspectiveTransform(np_raw_points, homography_matrix)
        transformed_service_boxes[box_name] = [tuple(p[0]) for p in transformed_points]
else:
    print("Warning: Court warp matrix not available. Cannot determine service boxes accurately for serve statistics.")

# Also transform the center service line
if court_detector.court_warp_matrix is not None and len(court_detector.court_warp_matrix) > 0 and hasattr(court_ref, 'middle_line'):
    # court_ref.middle_line is ((x1,y1), (x2,y2))
    # Use the already defined homography_matrix from service box transformation
    ref_center_line_pts = np.array([court_ref.middle_line[0], court_ref.middle_line[1]], dtype=np.float32).reshape((-1, 1, 2))
    transformed_center_line_pts = cv2.perspectiveTransform(ref_center_line_pts, homography_matrix)
    if transformed_center_line_pts is not None and len(transformed_center_line_pts) == 2:
        transformed_center_line = (tuple(transformed_center_line_pts[0][0]), tuple(transformed_center_line_pts[1][0]))
        # print(f"DEBUG: Transformed center line: {transformed_center_line}")
    else:
        print("Warning: Could not transform center line.")
else:
    print("Warning: Center line reference or homography not available for deuce/ad side determination.")

detection_model.find_player_2_box()

ball_landing_positions = []
transformed_court_boundary_points = []
net_line_coords_for_stats = []
ball_trajectories_for_net_miss = [] # Will hold the simplified single trajectory
serves_info_list = [] # To store info about each detected serve
rallies_data = []
all_detected_bounces_for_rallies = [] # For storing (x,y,frame_num) of bounces
transformed_service_boxes = {} # To store all service boxes transformed to image coordinates
# Variables to manage current point's serve state (will need more robust logic)
# current_server_id = None
# is_first_serve_in_point = True
# point_in_progress = False
first_serve_processed = False
transformed_center_line = None

# second part 
player1_boxes = detection_model.player_1_boxes
player2_boxes = detection_model.player_2_boxes

video = cv2.VideoCapture(input_video_path)
frame_i = 0

last = time.time() # start counting 
# while (True):
for img in frames:
    print('Tracking the ball: {}'.format(round( (currentFrame / total) * 100, 2)))
    frame_i += 1

    # detect the ball
    # img is the frame that TrackNet will predict the position
    # since we need to change the size and type of img, copy it to output_img
    output_img = img

    # resize it
    img = cv2.resize(img, (width, height))
    # input must be float type
    img = img.astype(np.float32)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    X = np.rollaxis(img, 2, 0)
    # prdict heatmap
    pr = m.predict(np.array([X]))[0]

    # since TrackNet output is ( net_output_height*model_output_width , n_classes )
    # so we need to reshape image as ( net_output_height, model_output_width , n_classes(depth) )
    pr = pr.reshape((height, width, n_classes)).argmax(axis=2)

    # cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
    pr = pr.astype(np.uint8)

    # reshape the image size as original input image
    heatmap = cv2.resize(pr, (output_width, output_height))

    # heatmap is converted into a binary image by threshold method.
    ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

    # find the circle in image with 2<=radius<=7
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                              maxRadius=7)


    output_img = mark_player_box(output_img, player1_boxes, currentFrame-1)
    output_img = mark_player_box(output_img, player2_boxes, currentFrame-1)
    
    PIL_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    PIL_image = Image.fromarray(PIL_image)

    # This entire block replaces the previous test serve logic.
    # It should be within the `for img in frames:` loop, after `output_img = img`.

    # global first_serve_processed # Not strictly needed as loop is top-level, but good for future refactor

    if not first_serve_processed and currentFrame > 10 and len(coords) > 5 and transformed_court_boundary_points and transformed_center_line:
        # Try to detect the very first serve of the video.

        initial_ball_pos = coords[0][:2] # (x,y) of the first detected ball coordinate
        server_id = None
        server_half = None # 'top' or 'bottom' half of the screen where server is

        # Determine which player is closer to the initial ball position AND near a baseline.
        # This requires player bounding boxes (player1_boxes, player2_boxes) to be available from the first pass.
        # For simplicity, we'll check if player boxes are defined and if the ball starts in their half.

        # Define baselines (approximate Y values from transformed court boundaries)
        # transformed_court_boundary_points: TL, TR, BR, BL
        baseline_y_top = (transformed_court_boundary_points[0][1] + transformed_court_boundary_points[1][1]) / 2
        baseline_y_bottom = (transformed_court_boundary_points[2][1] + transformed_court_boundary_points[3][1]) / 2
        court_center_y = (baseline_y_top + baseline_y_bottom) / 2

        # Check Player 1 (assuming player1_boxes means they are generally at the bottom)
        if detection_model.player_1_box is not None : # Check if player1_box is defined
            # Using detection_model.player_1_box which is [x,y,w,h]
            p1_box_center_y = detection_model.player_1_box[1] + detection_model.player_1_box[3] / 2
            if abs(p1_box_center_y - baseline_y_bottom) < output_height * 0.15 and initial_ball_pos[1] > court_center_y: # Player near bottom baseline, ball in bottom half
                server_id = "player1"
                server_half = "bottom"

        # Check Player 2 (assuming player2_boxes means they are generally at the top)
        if not server_id and detection_model.player_2_box is not None:
            p2_box_center_y = detection_model.player_2_box[1] + detection_model.player_2_box[3] / 2
            if abs(p2_box_center_y - baseline_y_top) < output_height * 0.15 and initial_ball_pos[1] < court_center_y: # Player near top baseline, ball in top half
                server_id = "player2"
                server_half = "top"

        if server_id:
            # Determine serving side (Deuce/Ad)
            center_line_avg_x = (transformed_center_line[0][0] + transformed_center_line[1][0]) / 2

            serving_from_deuce_side = False # Player's right side
            if server_half == "bottom":
                if initial_ball_pos[0] > center_line_avg_x:
                    serving_from_deuce_side = True
            elif server_half == "top":
                if initial_ball_pos[0] < center_line_avg_x:
                    serving_from_deuce_side = True

            target_box_name = None
            if server_half == "bottom":
                target_box_name = "top_left_box" if serving_from_deuce_side else "top_right_box"
            elif server_half == "top":
                target_box_name = "bottom_left_box" if serving_from_deuce_side else "bottom_right_box"

            actual_target_box_transformed = transformed_service_boxes.get(target_box_name)

            if actual_target_box_transformed:
                if bounce == 1 and 'idx' in locals() and idx is not None and len(idx) > 0 and \
                   'xy' in locals() and xy is not None and \
                   'coords_with_dummy_frame' in locals() and coords_with_dummy_frame is not None:

                    first_bounce_index_in_xy = idx[0]
                    if 0 <= first_bounce_index_in_xy < len(xy) and 0 <= first_bounce_index_in_xy < len(coords_with_dummy_frame):
                        landing_coord = tuple(xy[first_bounce_index_in_xy])
                        frame_of_bounce = coords_with_dummy_frame[first_bounce_index_in_xy][2]

                        serve_info = {
                            'landing_coord': landing_coord,
                            'is_first_serve': True,
                            'target_service_box_transformed_points': actual_target_box_transformed,
                            'frame_of_bounce': frame_of_bounce,
                            'server_id': server_id,
                            'served_from_deuce': serving_from_deuce_side
                        }
                        serves_info_list.append(serve_info)
                        first_serve_processed = True # Set flag
                        print(f"DEBUG: Processed first serve. Server: {server_id}, From Deuce: {serving_from_deuce_side}, Target Box: {target_box_name}, Landing: {landing_coord}")
            else:
                print(f"DEBUG: Could not determine target service box for first serve. Name: {target_box_name}")
        else:
            print("DEBUG: Could not identify server for the first serve.")

    # End of First Serve Detection Logic

    # check if there have any tennis be detected
    if circles is not None:
        # if only one tennis be detected
        if len(circles) == 1:

            x = int(circles[0][0][0])
            y = int(circles[0][0][1])

            coords.append([x,y])
            t.append(time.time()-last)

            # push x,y to queue
            q.appendleft([x, y])
            # pop x,y from queue
            q.pop()

        else:
            coords.append(None)
            t.append(time.time()-last)
            # push None to queue
            q.appendleft(None)
            # pop x,y from queue
            q.pop()

    else:
        coords.append(None)
        t.append(time.time()-last)
        # push None to queue
        q.appendleft(None)
        # pop x,y from queue
        q.pop()

    # draw current frame prediction and previous 7 frames as yellow circle, total: 8 frames
    for i in range(0, 8):
        if q[i] is not None:
            draw_x = q[i][0]
            draw_y = q[i][1]
            bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
            draw = ImageDraw.Draw(PIL_image)
            draw.ellipse(bbox, outline='yellow')
            del draw

    # Convert PIL image format back to opencv image format
    opencvImage = cv2.cvtColor(np.array(PIL_image), cv2.COLOR_RGB2BGR)

    output_video.write(opencvImage)

    # next frame
    currentFrame += 1

# everything is done, release the video
video.release()
output_video.release()

if minimap == 1:
  game_video = cv2.VideoCapture(output_video_path)

  fps1 = int(game_video.get(cv2.CAP_PROP_FPS))

  output_width = int(game_video.get(cv2.CAP_PROP_FRAME_WIDTH))
  output_height = int(game_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  print('game ', fps1)
  output_video = cv2.VideoWriter('VideoOutput/video_with_map.mp4', fourcc, fps, (output_width, output_height))
  
  print('Adding the mini-map...')

  # Remove Outliers 
  x, y = diff_xy(coords)
  remove_outliers(x, y, coords)
  # Interpolation
  coords = interpolation(coords)
  create_top_view(court_detector, detection_model, coords, fps)
  minimap_video = cv2.VideoCapture('VideoOutput/minimap.mp4')
  fps2 = int(minimap_video.get(cv2.CAP_PROP_FPS))
  print('minimap ', fps2)
  while True:
    ret, frame = game_video.read()
    ret2, img = minimap_video.read()
    if ret:
      output = merge(frame, img)
      output_video.write(output)
    else:
      break
  game_video.release()
  minimap_video.release()

output_video.release()

for _ in range(3):
  x, y = diff_xy(coords)
  remove_outliers(x, y, coords)

# interpolation
coords = interpolation(coords)

if coords: # coords is the list of [x,y] ball positions
    # For detect_net_misses, it expects a list of trajectories (shots).
    # As a temporary simplification, we treat the entire ball path as one "shot".
    # We also need to ensure each point in coords has at least (x,y,frame_number).
    # The current 'coords' list only has [x,y]. We'll add a dummy frame number for now.
    # THIS NEEDS TO BE REPLACED WITH ACTUAL SHOT SEGMENTATION LATER.
    coords_with_dummy_frame = [(pt[0], pt[1], i) for i, pt in enumerate(coords) if pt is not None]
    if coords_with_dummy_frame:
         ball_trajectories_for_net_miss = [coords_with_dummy_frame]

    # Prepare ball trajectory with frame numbers for shot segmentation
    # coords_with_dummy_frame is already [(x,y,frame_num), ...] from previous net miss setup.

    if coords_with_dummy_frame: # This is the full interpolated ball path with frame numbers
        print(f"DEBUG: Running shot segmentation on trajectory with {len(coords_with_dummy_frame)} points.")
        # Note: player_positions, court_info arguments for identify_rallies_and_shots are not fully used yet by it.
        # We'll pass what we have. Player positions would come from detection_model.
        # For court_info, we can pass the raw detected lines for now.

        # TODO: Get actual player positions if available and needed by rally_stats
        # player_positions_for_rally_stats = {} # e.g. {'player1': p1_coords_list, 'player2': p2_coords_list}

        # Pass necessary court boundary and net line info for future rally-ending logic
        # These should already be calculated by this point in the script

        rallies_data = identify_rallies_and_shots(
            ball_trajectory=coords_with_dummy_frame,
            player_positions=None, # Placeholder for actual player positions
            court_info=None, # Placeholder for more structured court info
            net_line_coords=net_line_coords_for_stats, # Already available
            court_boundary_points=transformed_court_boundary_points, # Already available
            all_detected_bounces=all_detected_bounces_for_rallies # Pass the new data
        )
        print(f"DEBUG: Shot segmentation resulted in {len(rallies_data)} rally/rallies.")
        if rallies_data and rallies_data[0]:
            print(f"DEBUG: First rally has {len(rallies_data[0])} shots.")

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
              ball_landing_positions.append(tuple(xy[i_bounce_idx]))

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
  i = 0
  while True:
    ret, frame = video.read()
    if ret:
      # if coords[i] is not None:
      if i in idx:
        center_coordinates = int(xy[i][0]), int(xy[i][1])
        radius = 3
        color = (255, 0, 0)
        thickness = -1
        cv2.circle(frame, center_coordinates, 10, color, thickness)
      i += 1
      output_video.write(frame)
    else:
      break

  video.release()
  output_video.release()

if rallies_data and net_line_coords_for_stats:
    all_shots_for_net_detection = []
    for rally in rallies_data:
        for shot in rally:
            all_shots_for_net_detection.append(shot['ball_path']) # shot['ball_path'] is list of (x,y,frame_num)

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
