import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import speech_recognition as sr
import pyaudio
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)

LEFT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]

LEFT_PUPIL = [474,475,476,477]
RIGHT_PUPIL =[469,470,471,472]

r = sr.Recognizer()


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1020)

with mp_face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,min_detection_confidence=0.9,min_tracking_confidence=0.9) as face_mesh:
    while True :
        ret, frame = cap.read()

        frame=cv2.flip(frame,1)
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img_h,img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
          mesh_points = np.array([np.multiply([p.x,p.y],[img_w,img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
          LEFT_IRIS = (mesh_points[LEFT_PUPIL[0]] + mesh_points[LEFT_PUPIL[1]] + mesh_points[LEFT_PUPIL[2]] + mesh_points[LEFT_PUPIL[3]])/4
          LEFT_IRIS_int_x = int(LEFT_IRIS[0])
          LEFT_IRIS_int_y = int(LEFT_IRIS[1])
          LEFT_IRIS_AVG = [LEFT_IRIS_int_x,LEFT_IRIS_int_y]

          RIGHT_IRIS = (mesh_points[RIGHT_PUPIL[0]] + mesh_points[RIGHT_PUPIL[1]] + mesh_points[RIGHT_PUPIL[2]] +
                       mesh_points[RIGHT_PUPIL[3]]) / 4
          RIGHT_IRIS_int_x = int(RIGHT_IRIS[0])
          RIGHT_IRIS_int_y = int(RIGHT_IRIS[1])
          RIGHT_IRIS_AVG = [RIGHT_IRIS_int_x, RIGHT_IRIS_int_y]

          CENTER_x = int((RIGHT_IRIS_int_x + LEFT_IRIS_int_x)/2)
          CENTER_y = int((RIGHT_IRIS_int_y+LEFT_IRIS_int_y)/2)
          CENTER_GAZE = [CENTER_x,CENTER_y]

          RIGHT_EYELID_UPPER = mesh_points[386]
          RIGHT_EYELID_LOWER = mesh_points[374]
          RIGHT_EYELID_DISTANCE = RIGHT_EYELID_LOWER[1]-RIGHT_EYELID_UPPER[1]



          LEFT_EYELID_UPPER = mesh_points[159]
          LEFT_EYELID_LOWER = mesh_points[145]
          LEFT_EYELID_DISTANCE = LEFT_EYELID_LOWER[1] - LEFT_EYELID_UPPER[1]


          cv2.polylines(frame,[mesh_points[LEFT_PUPIL]],True,(0,255,0),1,cv2.LINE_AA)
          cv2.polylines(frame, [mesh_points[RIGHT_PUPIL]], True, (0, 255, 0), 1, cv2.LINE_AA)

          cv2.circle(frame,(LEFT_IRIS_AVG),radius=1,color=(0,0,255),thickness=-1)
          cv2.circle(frame, (RIGHT_IRIS_AVG), radius=1, color=(0, 0, 255), thickness=-1)
          cv2.circle(frame, (CENTER_GAZE), radius=1, color=(255, 0, 0), thickness=-1)
          cv2.circle(frame, (RIGHT_EYELID_UPPER), radius=1, color=(255, 255, 255), thickness=-1)
          cv2.circle(frame, (LEFT_EYELID_UPPER), radius=1, color=(255, 255, 255), thickness=-1)
          cv2.circle(frame, (RIGHT_EYELID_LOWER), radius=1, color=(255, 255, 255), thickness=-1)
          cv2.circle(frame, (LEFT_EYELID_LOWER), radius=1, color=(255, 255, 255), thickness=-1)
          cv2.imshow('frame',frame)





          if RIGHT_EYELID_DISTANCE < 5 :
              print('right eye blinked')
              pyautogui.click(button='right')
              pyautogui.sleep(0.25)
          if LEFT_EYELID_DISTANCE < 5:

              print('left eye blinked')
              pyautogui.click(button='left')
              pyautogui.sleep(0.25)
          mouseGapX = int(((CENTER_GAZE[0]-700)/10)**3)
          mouseGapY = int(((CENTER_GAZE[1] - 400) / 10)**3)
          pyautogui.move(mouseGapX,mouseGapY)
          #print(mouseGapX)

       # with sr.Microphone() as source:
            #r.adjust_for_ambient_noise(source)
           # audio = r.listen(source)
           # try:
            #    voiceString = r.recognize_google(audio)
            #    print(voiceString)
           # except Exception as e:
            #    print("Error")



        if cv2.waitKey(1)& 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

