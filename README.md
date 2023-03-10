# AI Pose Trainer
AI Pose Trainer is an application where you can remotely use it and make confirmation whether your posture correct or not. This project is developed by using Mediapipe Model and OpenCv.

# Dependencies
### Install Mediapipe
    $ pip3 install mediapipe

### Libraries Used
    import cv2
    import numpy as np
    import time

# Mediapipe Pose Landmark Model (BlazePose GHUM 3D)
The landmark model in MediaPipe Pose predicts the location of 33 pose landmarks (see figure below).
![This is an image](/pose_tracking_full_body_landmarks.png)


# Posture Correction Output
### Bicep Curl
Please click to watch the output: [Bicep curl](/Posture%20Correction%20Output/bicep_curl.avi) 

### Shoulder Press
Please click to watch the output: [Shoulder press](/Posture%20Correction%20Output/shoulder_press.avi) 

# License
The MIT License (MIT). Please see [License File](/LICENSE) for more information.
