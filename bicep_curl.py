import cv2
import time
import numpy as np
from PoseModule import poseDetector
from PoseAnalysis import poseEvaluate

WINDOW_NAME = 'Pose Tracking'

# Full screen mode
cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

ptime = 0
detector = poseDetector()
poseEvaluate = poseEvaluate()

flag = 0 
counter = 0
stage = None
feedback = ''

countdown_timer = 5
hand_side = 'right_bicep_curl'

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    h, w = frame.shape[:2]

    # Calling the pose detection class method
    image= detector.findPose(frame, draw=False)
    
    if counter == 5 and hand_side =='right_bicep_curl':
        hand_side = 'left_bicep_curl'
        counter = 0
        countdown_timer = 5
    
    if counter == 5 and hand_side =='left_bicep_curl':
        hand_side = 'right_bicep_curl'
        counter = 0
        countdown_timer = 5   

    if detector.results.pose_landmarks:     
        if flag == 1:
            # Calling the call method of poseDetector class
            landmarks_r = detector.results.pose_landmarks.landmark
            mp_pose = detector.mp_pose.PoseLandmark

            # Bicep curl
            angle_e, angle_s, present = poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, hand_side) 

            # Curl counter logic with simply angle
            if 1.0 <= angle_s <=25.0 and angle_e <=85.0 and stage == 'down':
                # color = (0, 255, 0)
                stage = 'up'
                if present == 1:
                    counter +=1

            if 1.0 <= angle_s <=25.0 and 150.0 <= angle_e <= 175.0:
                # color = (255, 0, 255)
                stage = 'down' 

            # Feedback Logic
            color_err = (0, 0, 255)
            color_corr = (0, 255, 0)

            if angle_s > 25.0:
                poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, hand_side, color_err) 
                feedback = [0, 'Bad Posture!! Back Body Should be Straight or Upper Arm should be parallel to torso'] 
            elif angle_e > 175.0 and angle_e < 155.0:
                poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, hand_side, color_err) 
                feedback = [0, 'Bad Posture!! Form Arm pose gonne wrong']
            else:
                poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, hand_side, color_corr) 
                feedback = [1, 'Doing Good. Keep it up!!']  

            # Count curl
            cv2.putText(frame, 'Counter', (20, h-50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'{str(counter)} {stage}', (20, h-30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # Primary hand side
            cv2.putText(image, 'Bicep Curl: Right Hand' if hand_side == 'right_bicep_curl' else 'Bicep Curl: Left Hand', (w//2, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # Feedback
            if feedback[0] == 1:
                color = color_corr
            elif feedback[0] == 0:
                color = color_err
            elif present != 1:
                color = color_err

            cv2.putText(image, 'Pose Feedback', (10, 65), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            cv2.putText(image, feedback[1] if present == 1 else 'Arm and Hip are not detected properly', (10, 80), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)
           
    else:
        cv2.putText(image, 'Landmarks are not detected properly', (10, 80), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 0), 1)


    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(image, str(int(fps))+' fps', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    
    # Defined countdown
    if countdown_timer >=0:
        cv2.putText(image, 'Be Ready for Right Bicep Curl' if hand_side == 'right_bicep_curl' else 'Be Ready for Left Bicep Curl', (w//2-180, h//2-100), cv2.FONT_HERSHEY_SIMPLEX,0.8 , (0, 0, 255), 1)
        cv2.putText(image, str(countdown_timer), (w//2-35, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    
    cv2.imshow(WINDOW_NAME, image)

    # check the countdown
    if countdown_timer >=0:
        countdown_timer -= 1
        time.sleep(1)

    if countdown_timer == 0:
        flag = 1

    if cv2.waitKey(10) & 0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()      