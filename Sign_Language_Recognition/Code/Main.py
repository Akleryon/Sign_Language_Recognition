import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

model = pickle.load(open('/home/acraf/Bureau/VSCode/test/Code/model_file' + '.pkl', 'rb'))
font = cv2.FONT_HERSHEY_SIMPLEX

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image_height, image_width, _ = image.shape

    # Filter image to get only hand shape with green lines
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36, 25, 25), (70, 255,255))
    imask = mask>0
    image_to_process = np.zeros_like(image, np.uint8)
    image_to_process[imask] = image[imask]
    
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        points = []
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y)
        points.append(              
        hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y)
        points.append(
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z)
        points = np.reshape(points, (1, -1))
        
        test=tf.data.Dataset.from_tensor_slices(((points)))
        y_predict=model.predict(test.batch(1024))
        predict=np.argmax(y_predict,axis=-1)

        if predict == [0]:
          cv2.putText(image, 'Palm',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [1]:
          cv2.putText(image,'L',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [2]:
          cv2.putText(image,'Thumb Up',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [3]:
          cv2.putText(image,'Back End',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [4]:
          cv2.putText(image,'Hand Down',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [5]:
          cv2.putText(image,'Index',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [6]:
          cv2.putText(image,'OK',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [7]:
          cv2.putText(image,'KARATE',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [8]:
          cv2.putText(image,'C',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [9]:
          cv2.putText(image,'Back End with fingers',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [10]:
          cv2.putText(image,'Metal',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [11]:
          cv2.putText(image,'Brice de Nice',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [12]:
          cv2.putText(image,'Two',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [13]:
          cv2.putText(image,'Three',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)  
        elif predict == [14]:
          cv2.putText(image,'Four',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
        elif predict == [15]:
          cv2.putText(image,'Five',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)          
        else:
          cv2.putText(image,'IDK',(50, 50), 
                font, 1, 
                (255, 0, 0), 
                2, 
                cv2.LINE_4)
          
        mp_drawing.draw_landmarks(
          image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing.DrawingSpec(
            color=(0,255,0),
            thickness=3,
            circle_radius=5
          ),
          mp_drawing.DrawingSpec(
            color=(0,255,0),
            thickness=3,
            circle_radius=2
          )
        )
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
  cap.release()