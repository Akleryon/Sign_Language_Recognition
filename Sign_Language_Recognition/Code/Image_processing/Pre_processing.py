# Preprocess the images

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
df = pd.DataFrame(columns=['WRIST_x', 'WRIST_y', 'WRIST_z', 'THUMB_CMC_x', 'THUMB_CMC_y', 'THUMB_CMC_z', 'THUMB_MCP_x', 'THUMB_MCP_y', 'THUMB_MCP_z', 'THUMB_IP_x', 'THUMB_IP_y', 'THUMB_IP_z', 'THUMB_TIP_x', 'THUMB_TIP_y', 'THUMB_TIP_z', 'INDEX_FINGER_MCP_x', 'INDEX_FINGER_MCP_y', 'INDEX_FINGER_MCP_z', 'INDEX_FINGER_PIP_x', 'INDEX_FINGER_PIP_y', 'INDEX_FINGER_PIP_z', 'INDEX_FINGER_DIP_x', 'INDEX_FINGER_DIP_y', 'INDEX_FINGER_DIP_z', 'INDEX_FINGER_TIP_x', 'INDEX_FINGER_TIP_y', 'INDEX_FINGER_TIP_z', 'MIDDLE_FINGER_MCP_x', 'MIDDLE_FINGER_MCP_y', 'MIDDLE_FINGER_MCP_z', 'MIDDLE_FINGER_PIP_x', 'MIDDLE_FINGER_PIP_y', 'MIDDLE_FINGER_PIP_z', 'MIDDLE_FINGER_DIP_x', 'MIDDLE_FINGER_DIP_y', 'MIDDLE_FINGER_DIP_z', 'MIDDLE_FINGER_TIP_x', 'MIDDLE_FINGER_TIP_y', 'MIDDLE_FINGER_TIP_z', 'RING_FINGER_MCP_x', 'RING_FINGER_MCP_y', 'RING_FINGER_MCP_z', 'RING_FINGER_PIP_x', 'RING_FINGER_PIP_y', 'RING_FINGER_PIP_z', 'RING_FINGER_DIP_x', 'RING_FINGER_DIP_y', 'RING_FINGER_DIP_z', 'RING_FINGER_TIP_x', 'RING_FINGER_TIP_y', 'RING_FINGER_TIP_z', 'PINKY_MCP_x', 'PINKY_MCP_y', 'PINKY_MCP_z', 'PINKY_PIP_x', 'PINKY_PIP_y', 'PINKY_PIP_z', 'PINKY_DIP_x', 'PINKY_DIP_y', 'PINKY_DIP_z', 'PINKY_TIP_x', 'PINKY_TIP_y', 'PINKY_TIP_z'])

folder = ['test_pose/', 'train_pose/']
folder2 = ['01_palm/', '02_l/', '03_up/', '04_fist_moved/', '05_down/', '06_index/', '07_ok/', '08_palm_m/', '09_c/', '10_palm_u/', '11_heavy/', '12_hang/', '13_two/', '14_three/', '15_four/', '16_five/']
counter = 0
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for direction in folder:
      for direction2 in folder2:
        IMAGE_FILES = os.listdir('images/dataset/'+direction+direction2)
        for idx, file in enumerate(IMAGE_FILES):
          if file != '.DS_Store':
            # Read an image, flip it around y-axis for correct handedness output (see
            # above).
            image = cv2.flip(cv2.imread('images/dataset/'+direction+direction2+file), 1)
            # Convert the BGR image to RGB before processing.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print handedness and draw hand landmarks on the image.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
              continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
              points = []
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])
              points.append(              
              [hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y])
              points.append(
              [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z])
              
              df = df.append(pd.DataFrame(points), ignore_index = True)
              mp_drawing.draw_landmarks(
                  annotated_image,
                  hand_landmarks,
                  mp_hands.HAND_CONNECTIONS,
                  mp_drawing.DrawingSpec(
                    color=(0,255,0),
                    thickness=1,
                    circle_radius=2
                  ),
                  mp_drawing.DrawingSpec(
                    color=(0,255,0),
                    thickness=1,
                    circle_radius=1
                  )
                )
              
            hsv = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
            imask = mask>0
            final_img = np.zeros_like(annotated_image, np.uint8)
            final_img[imask] = annotated_image[imask]
            
            cv2.imwrite(          
            'images/processed/'+ direction2 + 'hand_' + str(counter) + '.jpeg', cv2.flip(final_img, 1))
            # Draw hand world landmarks.
            counter += 1 

compression_opts = dict(method='zip',
                        archive_name='out.csv')  
df.to_csv('out.zip', index=False,
          compression=compression_opts)
   