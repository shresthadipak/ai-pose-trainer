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
feedback = None

countdown_timer = 5

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    h, w = frame.shape[:2]

    # Calling the pose detection class method
    image= detector.findPose(frame, draw=False)

    if counter == 5:
        counter = 0
        countdown_timer = 10

    exercise = 'shoulder_press'
    if detector.results.pose_landmarks:     
        if flag == 1:
            # Calling the call method of poseDetector class
            landmarks_r = detector.results.pose_landmarks.landmark
            mp_pose = detector.mp_pose.PoseLandmark
            
            # Shoulder Press
            angle_er, angle_sr, angle_el, angle_sl, present = poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise)

            
            # Curl counter logic with angle
            if 65.0 <= angle_er <= 90.0 and 80.0 <= angle_sr <= 100.0 and 65.0 <= angle_el <= 90.0 and 80.0 <= angle_sl <= 100.0:
                # color = (0, 255, 0)
                stage = 'down'  
                
            if 160.0 <= angle_er <= 180.0 and 160.0 <= angle_sr <= 180.0 and 160.0 <= angle_el <= 180.0 and 160.0 <= angle_sl <= 180.0 and stage == 'down':
                # color = (255, 0, 255)
                stage = 'up'
                if present == 1:
                    counter +=1 

            # Feedback Logic
            color_err = (0, 0, 255)
            color_corr = (0, 255, 0)

            if stage == 'down':
                if  angle_er > 90.0 or angle_er < 65.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Left arm wrong posture!! Please make your fore arm straight so that it could make almost 90 degreee to upper arm!!!'
                elif angle_el > 90.0 or angle_el < 65.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Right arm wrong posture!! Please make your fore arm straight so that it could make almost 90 degreee to upper arm!!!'  
                elif angle_er > 90.0 or angle_er < 65.0 or angle_el > 90.0 or angle_el < 65.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Both arm wrong posture!! Please make your fore arm straight so that it could make almost 90 degreee to upper arm!!!' 
                elif angle_sr > 100.0 or angle_sr < 80.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Right upper arm  tilted make it proper position so that it could make almost 90 degree to torso'
                elif angle_sl > 100.0 or angle_sl < 80.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Right upper arm  tilted make it proper position so that it could make almost 90 degree to torso'    
                else:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_corr)
                    feedback = 'Doing Good. Keep it up!!'  

            elif stage == 'up':        
                if  angle_er > 180.0 or angle_er < 160.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Left arm wrong posture!! Please make your fore arm straight so that it could make almost 180 degreee to upper arm!!!'
                elif angle_el > 180.0 or angle_el < 160.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Right arm wrong posture!! Please make your fore arm straight so that it could make almost 180 degreee to upper arm!!!'  
                elif angle_er > 180.0 or angle_er < 160.0 or angle_el > 180.0 or angle_el < 160.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Both arm wrong posture!! Please make your fore arm straight so that it could make almost 180 degreee to upper arm!!!' 
                elif angle_sr > 180.0 or angle_sr < 160.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Right upper arm  tilted make it proper position so that it could make almost 180 degree to torso'
                elif angle_sl > 180.0 or angle_sl < 160.0:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_err)
                    feedback = 'Right upper arm  tilted make it proper position so that it could make almost 180 degree to torso'    
                else:
                    poseEvaluate.evaluate_pose(image, landmarks_r, mp_pose, exercise, color_corr)
                    feedback = 'Doing Good. Keep it up!!'
               
                
            # Count curl
            cv2.putText(frame, 'Counter', (20, h-50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'{str(counter)} {stage}', (20, h-30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # Primary hand side
            cv2.putText(image, 'Shoulder Excersie: Shoulder Press', (w//2, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            # Feedback
            if feedback[0] == 1:
                color = color_corr
            elif feedback[0] == 0:
                color = color_err
            elif present != 1:
                color = color_err

            cv2.putText(image, 'Pose Feedback', (10, 65), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
            cv2.putText(image, feedback if present == 1 else 'Arm and Hip are not detected properly', (10, 80), cv2.FONT_HERSHEY_PLAIN, 0.7, color, 1)
            
        
        if counter == 5:
            flag = 0

        if flag == 0 and counter == 5:
            p_no = 75
                  

    else:
        cv2.putText(image, 'Error Message:', (10, 65), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.putText(image, 'Landmarks are not detected properly', (10, 80), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 1)


    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(image, str(int(fps))+' fps', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    
    # Defined countdown
    if countdown_timer >=0:
        cv2.putText(image, 'Be Ready for Shoulder Press', (w//2-180, h//2-100), cv2.FONT_HERSHEY_SIMPLEX,0.8 , (0, 0, 255), 1)
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
