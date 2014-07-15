import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.animation as animation

# statics
# length of buffer for smoothing
BUFFER_LENGTH = 30

# threshold for navigation vs stopping
THRESHOLD = 1.5

cap = cv2.VideoCapture("/home/dan/hyperlayer/opticalflow/IMG_0067.MOV")
#cap = cv2.VideoCapture("169.254.220.174/mjpg/video.mjpg")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

width = cap.get(3)
height = cap.get(4)

print width
print height

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

#frameskip
frameSkip = 0

#setup histogram
histogram = np.zeros(50, dtype = np.float32)

#variance buffer
varianceBuffer = []

#for rendering plot
plot = np.zeros_like(old_frame)

while(1):
    #frameskip
    for i in range(frameSkip):
        ret,frame = cap.read()

    #clear vector visualization
    vectorField = np.zeros_like(old_frame)
    plot = np.zeros_like(old_frame)
    
    #store vector magnitudes
    magnitudes = []
    
    ret,frame = cap.read()  
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # check if features should be recalculated
    if len(good_new) < 5:
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        #when redrawing throw away old features
        p1 = p0
        continue
        
    # draw the tracks
    '''
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(frame,(a,b),5,(0,0,255),-1)
        #draw vector field
        cv2.circle(vectorField,(c,d),1,(0,255,0),-1)
        cv2.line(vectorField, (a,b), (c,d), (0,255,0), 1)
    
    img = cv2.add(frame,mask)
    '''
    
    for i,(new,old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        
        #draw vector field
        cv2.circle(vectorField,(c,d),1,(0,255,0),-1)
        cv2.line(vectorField, (a,b), (c,d), (0,255,0), 1)
        
        #add data to calculate variance
        mag = np.sqrt((a-c) ** 2 + (b-d) ** 2)
        magnitudes.append(mag)
    
    width = 300
    for index, eachVariance in enumerate(varianceBuffer):
        cv2.circle(plot, (index, int(eachVariance)), 1, (50,50,255), -1)
    
    variance = np.var(magnitudes)
    cv2.putText(vectorField, 'magnitude variance: %s' % variance, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    
    if len(varianceBuffer) > BUFFER_LENGTH:
        varianceMean = np.mean(varianceBuffer[-BUFFER_LENGTH:])
        #if variance > 1 and varianceMean > 1:
        if varianceMean > THRESHOLD:
            cv2.putText(vectorField, 'navigating', (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
        #elif variance < 1 and varianceMean < 1:
        elif varianceMean < THRESHOLD:
            cv2.putText(vectorField, 'stopped', (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
        else:
            cv2.putText(vectorField, 'thinking...', (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255)) 
    else:
        if variance > 1:
            cv2.putText(vectorField, 'navigating', (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
        else:
            cv2.putText(vectorField, 'stopped', (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
    
    varianceBuffer.append(variance)        
    img = frame
    cv2.imshow('img',img)
    cv2.imshow('vectorField', vectorField)
    cv2.imshow('plot', plot)
    
    #move window
    cv2.moveWindow('img', 0, 0)
    cv2.moveWindow('vectorField', 500, 0)
    cv2.moveWindow('plot', 0, 400)
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    '''
    # check if features should be recalculated
    if len(good_new) < 5:
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        #when redrawing throw away old features
        p1 = p0
    '''
    if cv2.waitKey(25) == 27:
        break

cv2.destroyAllWindows()
cap.release()

