import cv2
import numpy as np

cap = cv2.VideoCapture("169.254.220.174/mjpg/video.mjpg")

while(1):
    #frameskip
    ret,frame = cap.read()
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(25) == 27:
        break
