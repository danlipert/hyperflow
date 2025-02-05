import cv2
from numpy import *

def draw_flow(im,flow,step=16):
    h,w = im.shape[:2]
    y,x = mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
    fx,fy = flow[y,x].T

    # create line endpoints
    lines = vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
    lines = int32(lines)

    # create image and draw
    #vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    vis = im
    for (x1,y1),(x2,y2) in lines:
        cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
    return vis


cap = cv2.VideoCapture('/home/dan/hyperlayer/opticalflow/IMG_0060.MOV')

ret,im = cap.read()
prev_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

while True:
    # get grayscale image
    ret,im = cap.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # compute flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray,gray,0.5,1,3,15,3,5,1)
    prev_gray = gray
    
    if cv2.waitKey(25) == 27:
        break
        
    flowim = draw_flow(im, flow)
    cv2.imshow('video',im)
    cv2.imshow('flow', flowim)
