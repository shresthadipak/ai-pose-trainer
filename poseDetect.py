import cv2
from PoseModule import poseDetector

detector = poseDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    image = detector.findPose(frame, draw=True) 

    cv2.imshow("Pose Detection", image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()