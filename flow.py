import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.animation as animation
from collections import deque
from datetime import timedelta


# this was an idea, but somewhere along the path of implementation I forgot what it was. I'm leaving this in case I remember...
class FrameQueue(deque):
    def __init__(self, maxlen=10):
        self.maxlength = maxlen

    def full(self):
        if len(self) == self.maxlength:
            full = True
        else:
            full = False
        return full

    def addFrame(self, frame):
        if self.full():
            self.popleft()
        self.append(frame)

    def newest(self):
        return self[len(self) - 1]


# statics
# length of buffer for smoothing
BUFFER_LENGTH = 15

# threshold for navigation vs stopping
THRESHOLD = 0.8
LOOKING_THRESHOLD = 2

# number of corners to detect
MAX_FEATURES = 100
MIN_FEATURES = 5
RECALC_PERCENTAGE = 0.25

# option to display images to screen
DISPLAY = False

# window names
VIDEO_WINDOW = 'img'
VECTOR_WINDOW = 'vectorField'
PLOT_WINDOW = 'plot'

invideofile = "/Users/phoetrymaster/Hyperlayer/hyperflow/IMG_2899.MOV"
outdir = "/Users/phoetrymaster/Hyperlayer/hyperflow/"

filename, ext = os.path.splitext(os.path.basename(invideofile))
outvideofile = os.path.join(os.path.dirname(invideofile), filename + "_processed.mov")

cap = cv2.VideoCapture(invideofile)
#cap = cv2.VideoCapture("169.254.220.174/mjpg/video.mjpg")

cv_fourcc_code = cap.get(6)
frame_rate = cap.get(5)
frame_height = cap.get(4)
frame_width = cap.get(3)
video_length = timedelta(seconds=(cap.get(7) * (1 / frame_rate)))

fourcc = cv2.cv.CV_FOURCC(*'mp4v')
print(fourcc, cv_fourcc_code, frame_height, frame_width, frame_rate)
writer = cv2.VideoWriter(outvideofile, fourcc, frame_rate, (int(frame_width), int(frame_height)), True)

# try is to allow finally statement at end to always close files properly
try:
    # create windows if display is selected
    if DISPLAY:
        # create windows
        cv2.namedWindow(VIDEO_WINDOW, flags=cv2.WINDOW_NORMAL)
        cv2.namedWindow(VECTOR_WINDOW, flags=cv2.WINDOW_NORMAL)
        cv2.namedWindow(PLOT_WINDOW, flags=cv2.WINDOW_NORMAL)

        # move windows
        cv2.moveWindow(VIDEO_WINDOW, 0, 0)
        cv2.moveWindow(VECTOR_WINDOW, 500, 0)
        cv2.moveWindow(PLOT_WINDOW, 0, 400)

        #resize window
        cv2.resizeWindow(VIDEO_WINDOW, 500, int(frame_width / 500 * frame_height))
        cv2.resizeWindow(VECTOR_WINDOW, 500, int(frame_width / 500 * frame_height))
        cv2.resizeWindow(PLOT_WINDOW, 500, int(frame_width / 500 * frame_height))

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = MAX_FEATURES,
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
    histogram = np.zeros(MAX_FEATURES, dtype = np.float32)

    #variance buffer
    varianceBuffer = []
    movementBuffer = []

    #for rendering plot
    plot = np.zeros_like(old_frame)

    while(1):
        timestamp = timedelta(seconds=(cap.get(0) / 1000))
        sys.stdout.write("Processed {0} of {1}\r".format(timestamp, video_length))
        sys.stdout.flush()
        newFeatures = False
        #frameskip
        for i in range(frameSkip):
          ret,frame = cap.read()


        #clear vector visualization
        vectorField = np.zeros_like(old_frame)
        plot = np.zeros_like(old_frame)

        #store vector magnitudes
        magnitudes = []

        ret, frame = cap.read()

        if ret==True:

          frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

          # calculate optical flow
          p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

          # Select good points
          good_new = p1[st==1]
          good_old = p0[st==1]

          # check if features should be recalculated
          if len(good_new) < MIN_FEATURES or len(good_new) < (RECALC_PERCENTAGE * len(p1)):
              old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
              p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
              # calculate optical flow
              p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
              #when redrawing throw away old features
              #p1 = p0
              #newFeatures = True
              good_new = p1[st==1]
              good_old = p0[st==1]
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

          histogramLeftRight = np.zeros(50, dtype = np.float32)
          histogramUpDown    = np.zeros(50, dtype = np.float32)
          for i,(new,old) in enumerate(zip(good_new, good_old)):
              a,b = new.ravel()
              c,d = old.ravel()

              #draw vector field
              cv2.circle(vectorField,(c,d),1,(0,255,0),-1)
              cv2.line(vectorField, (a,b), (c,d), (0,255,0), 1)

              #add data to calculate variance
              mag = np.sqrt((a-c) ** 2 + (b-d) ** 2)
              magnitudes.append(mag)

              #calculate direction of feature movement
              histogram[i] = a - c

          width = 300
          for index, eachVariance in enumerate(varianceBuffer):
              cv2.circle(plot, (index, int(eachVariance)), 1, (50,50,255), -1)

          variance = np.var(magnitudes)
          cv2.putText(vectorField, 'magnitude variance: %s' % variance, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

          if newFeatures is False:
              movement = np.mean(histogram)
              cv2.putText(vectorField, 'movement: %s' % movement, (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
          else:
              movement = 0
              cv2.putText(vectorField, 'movement: Recalculating...', (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

          if len(varianceBuffer) > BUFFER_LENGTH:
              varianceMean = np.mean(varianceBuffer[-BUFFER_LENGTH:])
              movementMean = np.mean(movementBuffer[-BUFFER_LENGTH:])
              #if variance > 1 and varianceMean > 1:
              if varianceMean > THRESHOLD:
                  cv2.putText(vectorField, 'navigating', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
              #elif varianceMean > THRESHOLD and movementMean > LOOKING_THRESHOLD:
                  #cv2.putText(vectorField, 'looking', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
              #elif variance < 1 and varianceMean < 1:
              elif varianceMean < THRESHOLD and abs(movementMean) < LOOKING_THRESHOLD:
                  cv2.putText(vectorField, 'stopped', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
              elif varianceMean < THRESHOLD and abs(movementMean) > LOOKING_THRESHOLD:
                  cv2.putText(vectorField, 'looking', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))
              else:
                  cv2.putText(vectorField, 'thinking...', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255))
          else:
              if variance > 1:
                  cv2.putText(vectorField, 'navigating', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
              else:
                  cv2.putText(vectorField, 'stopped', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))

          # append values to buffers
          varianceBuffer.append(variance)
          movementBuffer.append(movement)

          # write frame to output
          writer.write(vectorField)

          if DISPLAY:
              img = frame
              # display to screen
              cv2.imshow(VIDEO_WINDOW,img)
              cv2.imshow(VECTOR_WINDOW, vectorField)
              cv2.imshow(PLOT_WINDOW, plot)

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
        else:
          break
except Exception as e:
    import traceback
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print e
    traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
finally:
    cv2.destroyAllWindows()
    cap.release()
    writer.release()
