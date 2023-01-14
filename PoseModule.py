import cv2
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

class poseDetector():

    def __init__(self, detectionConf=0.5, 
                 trackingConf=0.5):

        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        # Initialize the pipeline
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # pose tracking on webcam
        self.pose = self.mp_pose.Pose(min_detection_confidence = self.detectionConf, 
                                        min_tracking_confidence = self.trackingConf)

    def findPose(self, img, draw=True):   
        #convert frame to rgb
        img.flags.writeable = False
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.pose.process(imgRGB)

        # Render detection
        if self.results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(image = img,
                                        landmark_list = self.results.pose_landmarks,
                                        connections = self.mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec = self.mp_drawing.DrawingSpec(color=(245, 117, 60), thickness=2, circle_radius=2),
                                        connection_drawing_spec = self.mp_drawing.DrawingSpec(color =(245, 66, 230), thickness=2, circle_radius=2)
                                        )
        
        return img
 
        
def main():
    poseDetector()

if __name__ == "__main__":
    main()
    

