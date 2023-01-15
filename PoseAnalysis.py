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