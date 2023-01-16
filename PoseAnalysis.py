import numpy as np
import cv2


class calculateAngle():

    def __init__(self):
        pass
        
    def angleFinder(landmark1, landmark2, landmark3):
        landmark1 = np.array(landmark1)
        landmark2 = np.array(landmark2)
        landmark3 = np.array(landmark3)
        
        radians = np.arctan2(landmark3[1] - landmark2[1], landmark3[0]- landmark2[0]) - np.arctan2(landmark1[1] - landmark2[1], landmark1[0] - landmark2[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle

        return angle


class poseEvaluate():

    def __init__(self, rbicep='right_bicep_curl', lbicep='left_bicep_curl', shoulder='shoulder_press'):
        self.right_bicep_curl = rbicep
        self.left_bicep_curl = lbicep
        self.shoulder_press = shoulder

    
    def evaluate_pose(self, image, landmarks, mp_pose, exercise, color=None):
        if exercise == self.right_bicep_curl:
            return self._right_bicep_curl(image, landmarks, mp_pose, color)
        elif exercise == self.left_bicep_curl: 
            return self._left_bicep_curl(image, landmarks, mp_pose, color)  
        elif exercise == self.shoulder_press:
            return self._shoulder_press(image, landmarks, mp_pose, color)     
        else:
            print('Exercise is not recognized')  

    
    def _left_bicep_curl(self, image, landmarks, mp_pose, color=None, draw=True):
        self.h, self. w, _ = image.shape
        # Get the coordinates of Right part
        self.rhip = [landmarks[mp_pose.RIGHT_HIP.value].x, landmarks[mp_pose.RIGHT_HIP.value].y]
        self.rshoulder = [landmarks[mp_pose.RIGHT_SHOULDER.value].x, landmarks[mp_pose.RIGHT_SHOULDER.value].y]
        self.relbow = [landmarks[mp_pose.RIGHT_ELBOW.value].x, landmarks[mp_pose.RIGHT_ELBOW.value].y]
        self.rwrist = [landmarks[mp_pose.RIGHT_WRIST.value].x, landmarks[mp_pose.RIGHT_WRIST.value].y]
        self.lshoulder = [landmarks[mp_pose.LEFT_SHOULDER.value].x, landmarks[mp_pose.LEFT_SHOULDER.value].y]

        # angle right part
        angle_er = calculateAngle.angleFinder(self.rshoulder, self.relbow, self.rwrist)
        angle_sr = calculateAngle.angleFinder(self.rhip, self.rshoulder, self.relbow)


        # Visiblity Check
        self.rHip_conf = landmarks[mp_pose.RIGHT_HIP.value].visibility
        self.rShoulder_conf = landmarks[mp_pose.RIGHT_SHOULDER.value].visibility
        self.rElbow_conf = landmarks[mp_pose.RIGHT_ELBOW.value].visibility
        self.rWrist_conf = landmarks[mp_pose.RIGHT_WRIST.value].visibility

        # print(self.lWrist_conf)

        present = 0
        if self.rHip_conf > 0.1 and  self.rShoulder_conf > 0.1 and  self.rElbow_conf > 0.1 and self.rWrist_conf > 0.1:
            present = 1 

        
        if color:
            color = color
        else:
            color = (255, 0, 255)

        # print(color)    

        if draw:
            cv2.circle(image, tuple(np.multiply(self.relbow, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.relbow, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.rshoulder, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.rshoulder, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.rwrist, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.rwrist, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.rhip, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.rhip, [self.w, self.h]).astype(int)), 15, color, 2)

            # cv2.putText(image, str(int(angle_er)), tuple(np.multiply(self.relbow, [self.w, self.h]).astype(int)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            # cv2.putText(image, str(int(angle_sr)), tuple(np.multiply(self.rshoulder, [self.w, self.h]).astype(int)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            
            return angle_er, angle_sr, present
             
            
    def _right_bicep_curl(self, image, landmarks, mp_pose, color=None, draw=True):
        self.h, self. w, _ = image.shape
        # Get the coordinates of left part
        self.lhip = [landmarks[mp_pose.LEFT_HIP.value].x, landmarks[mp_pose.LEFT_HIP.value].y]
        self.lshoulder = [landmarks[mp_pose.LEFT_SHOULDER.value].x, landmarks[mp_pose.LEFT_SHOULDER.value].y]
        self.lelbow = [landmarks[mp_pose.LEFT_ELBOW.value].x, landmarks[mp_pose.LEFT_ELBOW.value].y]
        self.lwrist = [landmarks[mp_pose.LEFT_WRIST.value].x, landmarks[mp_pose.LEFT_WRIST.value].y]
        self.rshoulder = [landmarks[mp_pose.RIGHT_SHOULDER.value].x, landmarks[mp_pose.RIGHT_SHOULDER.value].y]
        
        # angle left part
        angle_el = calculateAngle.angleFinder(self.lshoulder, self.lelbow, self.lwrist)
        angle_sl = calculateAngle.angleFinder(self.lhip, self.lshoulder, self.lelbow) 

        # Visiblity Check
        self.lHip_conf = landmarks[mp_pose.LEFT_HIP.value].visibility
        self.lShoulder_conf = landmarks[mp_pose.LEFT_SHOULDER.value].visibility
        self.lElbow_conf = landmarks[mp_pose.LEFT_ELBOW.value].visibility
        self.lWrist_conf = landmarks[mp_pose.LEFT_WRIST.value].visibility

        # print(self.lWrist_conf)

        present = 0
        if self.lHip_conf > 0.1 and  self.lShoulder_conf > 0.1 and  self.lElbow_conf > 0.1 and self.lWrist_conf > 0.1:
            present = 1 
 

        if color:
            color = color
        else:
            color = (255, 0, 255)

        if draw:
            cv2.circle(image, tuple(np.multiply(self.lelbow, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.lelbow, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.lshoulder, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.lshoulder, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.lwrist, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.lwrist, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.lhip, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.lhip, [self.w, self.h]).astype(int)), 15, color, 2)
            # cv2.circle(image, tuple(np.multiply(self.neck, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            # cv2.circle(image, tuple(np.multiply(self.neck, [self.w, self.h]).astype(int)), 15, color, 2)


        return angle_el, angle_sl, present    

    def _shoulder_press(self, image, landmarks, mp_pose, color=None, draw=True):
        self.h, self. w, _ = image.shape
        # Get the coordinates of Right part
        self.rhip = [landmarks[mp_pose.RIGHT_HIP.value].x, landmarks[mp_pose.RIGHT_HIP.value].y]
        self.rshoulder = [landmarks[mp_pose.RIGHT_SHOULDER.value].x, landmarks[mp_pose.RIGHT_SHOULDER.value].y]
        self.relbow = [landmarks[mp_pose.RIGHT_ELBOW.value].x, landmarks[mp_pose.RIGHT_ELBOW.value].y]
        self.rwrist = [landmarks[mp_pose.RIGHT_WRIST.value].x, landmarks[mp_pose.RIGHT_WRIST.value].y]

        # Get the coordinates of left part
        self.lhip = [landmarks[mp_pose.LEFT_HIP.value].x, landmarks[mp_pose.LEFT_HIP.value].y]
        self.lshoulder = [landmarks[mp_pose.LEFT_SHOULDER.value].x, landmarks[mp_pose.LEFT_SHOULDER.value].y]
        self.lelbow = [landmarks[mp_pose.LEFT_ELBOW.value].x, landmarks[mp_pose.LEFT_ELBOW.value].y]
        self.lwrist = [landmarks[mp_pose.LEFT_WRIST.value].x, landmarks[mp_pose.LEFT_WRIST.value].y]

        # angle left part
        angle_el = calculateAngle.angleFinder(self.lshoulder, self.lelbow, self.lwrist)
        angle_sl = calculateAngle.angleFinder(self.lhip, self.lshoulder, self.lelbow)

        # angle right part
        angle_er = calculateAngle.angleFinder(self.rshoulder, self.relbow, self.rwrist)
        angle_sr = calculateAngle.angleFinder(self.rhip, self.rshoulder, self.relbow)


        # Visibilty Check

        # Visiblity Check of left part
        self.lHip_conf = landmarks[mp_pose.LEFT_HIP.value].visibility
        self.lShoulder_conf = landmarks[mp_pose.LEFT_SHOULDER.value].visibility
        self.lElbow_conf = landmarks[mp_pose.LEFT_ELBOW.value].visibility
        self.lWrist_conf = landmarks[mp_pose.LEFT_WRIST.value].visibility

        # Visiblity Check of right part
        self.rHip_conf = landmarks[mp_pose.RIGHT_HIP.value].visibility
        self.rShoulder_conf = landmarks[mp_pose.RIGHT_SHOULDER.value].visibility
        self.rElbow_conf = landmarks[mp_pose.RIGHT_ELBOW.value].visibility
        self.rWrist_conf = landmarks[mp_pose.RIGHT_WRIST.value].visibility

        present = 0
        if self.rHip_conf > 0.1 and  self.rShoulder_conf > 0.1 and  self.rElbow_conf > 0.1 and self.rWrist_conf > 0.1 and self.lHip_conf > 0.1 and  self.lShoulder_conf > 0.1 and  self.lElbow_conf > 0.1 and self.lWrist_conf > 0.1:
            present = 1 


        # Draw the keypoints
        if color:
            color = color
        else:
            color = (255, 0, 255)

        if draw:
            # Right part
            cv2.circle(image, tuple(np.multiply(self.relbow, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.relbow, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.rshoulder, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.rshoulder, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.rwrist, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.rwrist, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.rhip, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.rhip, [self.w, self.h]).astype(int)), 15, color, 2)

            # Left part
            cv2.circle(image, tuple(np.multiply(self.lelbow, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.lelbow, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.lshoulder, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.lshoulder, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.lwrist, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.lwrist, [self.w, self.h]).astype(int)), 15, color, 2)
            cv2.circle(image, tuple(np.multiply(self.lhip, [self.w, self.h]).astype(int)), 3, color, cv2.FILLED)
            cv2.circle(image, tuple(np.multiply(self.lhip, [self.w, self.h]).astype(int)), 15, color, 2)

            # cv2.putText(image, str(int(angle_er)), tuple(np.multiply(self.relbow, [self.w-200, self.h]).astype(int)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            # cv2.putText(image, str(int(angle_sr)), tuple(np.multiply(self.rshoulder, [self.w-200, self.h]).astype(int)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

        return  angle_er, angle_sr, angle_el, angle_sl, present   
    