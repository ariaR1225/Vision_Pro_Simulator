import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import glob

import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image

from utils import CvFpsCalc
from utils.util import *
from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--mirror",
                        help='mirror the camera feed',
                        type=int,
                        default=1)
    args = parser.parse_args()

    return args


def main():
    # argument parsing 
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    mirror = args.mirror
    print(mirror)

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # load images for photo gallery
    image_files = sorted(glob.glob('asset/img*.jpg'))
    overlay_images = []
    desired_width = 200
    desired_height = 200
    
    for img_path in image_files:
        img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (desired_width, desired_height))
        if img.shape[2] == 3: 
            img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
  
        alpha = 128  
        img[:, :, 3] = alpha
        
        overlay_images.append(img)
    
    current_image_index = 0
    current_angle = 0
    current_scale = 1.
    finger_counter = 0
    last_gesture = None
    
    # load model weights
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS 
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # deque init
    history_length = 16
    point_history = deque(maxlen=history_length)
    drag_history = deque(maxlen=history_length)
    paint_history = deque(maxlen=128)
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0

    # app starts
    while True:
        fps = cvFpsCalc.get()

        # to end app
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # camera reading
        ret, image = cap.read()
        if not ret:
            break
        if mirror:
            image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # hand detection
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                results.multi_handedness):
                # bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)                
                # write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 0:
                    paint_history.clear()

                if hand_sign_id == 1:  # Pinch
                    drag_history.append(landmark_list[8])
                else:
                    drag_history.append([0, 0])

                if hand_sign_id == 2:  # Pointer 
                    point_history.append(landmark_list[8])
                    paint_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])
                    paint_history.append([0, 0])

                # finger motion classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # calculate the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # check for consecutive gestures
                current_finger_motion = most_common_fg_id[0][0]
                
                if current_finger_motion == last_gesture:
                    finger_counter += 1
                else:
                    finger_counter = 1
                
                last_gesture = current_finger_motion
                
                # change image if consecutive gestures detected
                if finger_counter >= 15:

                    if current_finger_motion == 3:
                        # to next image
                        current_image_index = (current_image_index + 1) % len(overlay_images)
                        finger_counter = 0
                    if current_finger_motion == 4:
                        # previous image
                        current_image_index = (current_image_index - 1) % len(overlay_images)
                        finger_counter = 0

                    if current_finger_motion == 1:  # clockwise rotation
                        current_angle = (current_angle - 15) % 360
                        finger_counter = 0
                    if current_finger_motion == 2:  # counter-clockwise rotation
                        current_angle = (current_angle + 15) % 360
                        finger_counter = 0

                    if current_finger_motion == 5:  # zoom out
                        current_scale =  max(current_scale - 0.1, 0.2)
                        finger_counter = 0
                    if current_finger_motion == 6:  # zoom in
                        current_scale = min(current_scale + 0.1, 3.0)
                        finger_counter = 0

                # drawing
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])
            drag_history.append([0, 0])
            paint_history.append([0, 0])
        
        debug_image = draw_info(debug_image, fps, mode, number)
        # switch mode
        if mode == 0: # Photos
            debug_image = drag_img(debug_image, overlay_images[current_image_index], drag_history, current_angle, current_scale)
            debug_image = draw_point_history(debug_image, point_history)
        if mode == 1: # Canvas
            debug_image = draw_paint_history(debug_image, paint_history)
        if mode == 2: # Data collection (hand gesture)
            debug_image = draw_landmarks(debug_image, landmark_list)
        if mode == 3: # Data collection (finger motion)
            debug_image = draw_point_history(debug_image, point_history)

        cv.imshow('Vision Pro Simulator', debug_image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()