import os
import sys
import cv2
import numpy as np
import scipy.stats as stats
from collections import deque
from datetime import timedelta
import datetime as dt
import warnings

warnings.filterwarnings('error')

class NoFeatures(Exception):
    pass

def extract_capture_metadata(cap):
    '''
    extracts metadata on framerate, resolution, codec, and length from opencv video capture object
    '''
    cv_fourcc_code = cap.get(6)
    FRAME_RATE = cap.get(5)
    FRAME_HEIGHT = cap.get(4)
    FRAME_WIDTH = cap.get(3)
    VIDEO_LENGTH = timedelta(seconds=(cap.get(7) * (1 / FRAME_RATE)))
    return (cv_fourcc_code, FRAME_RATE, FRAME_HEIGHT, FRAME_WIDTH, VIDEO_LENGTH)

def setup_display_windows():
    '''
    create and configure live display windows
    '''
    cv2.namedWindow(VIDEO_WINDOW, flags=cv2.WINDOW_NORMAL)
    cv2.namedWindow(VECTOR_WINDOW, flags=cv2.WINDOW_NORMAL)
    cv2.namedWindow(PLOT_WINDOW, flags=cv2.WINDOW_NORMAL)

    # move windows
    cv2.moveWindow(VIDEO_WINDOW, 0, 0)
    cv2.moveWindow(VECTOR_WINDOW, 500, 0)
    cv2.moveWindow(PLOT_WINDOW, 0, 400)

    #resize window
    cv2.resizeWindow(VIDEO_WINDOW, 500, int(FRAME_WIDTH / 500 * FRAME_HEIGHT))
    cv2.resizeWindow(VECTOR_WINDOW, 500, int(FRAME_WIDTH / 500 * FRAME_HEIGHT))
    cv2.resizeWindow(PLOT_WINDOW, 500, int(FRAME_WIDTH / 500 * FRAME_HEIGHT))

def draw_text_overlay(vectorField,
                      varianceLRmean, varianceUDmean, movementLRmean, movementUDmean,
                      varianceLR, movementLR, varianceUD, movementUD):
    '''
    draws text overlay on vector field window
    '''

    cv2.putText(vectorField, 'variance L-R: %s, %s' % (varianceLR, varianceLRmean), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(vectorField, 'movement L-R: %s, %s' % (movementLR, movementLRmean), (0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(vectorField, 'variance U-D: %s, %s' % (varianceUD, varianceUDmean), (0,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))
    cv2.putText(vectorField, 'movement U-D: %s, %s' % (movementUD, movementUDmean), (0,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0))

    if varianceLRmean > HIGH_VARIANCE_LR or varianceUDmean > HIGH_VARIANCE_UD:
        cv2.putText(vectorField, 'navigating', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
    elif (varianceLRmean < MEDIUM_VARIANCE_LR or varianceUDmean < MEDIUM_VARIANCE_UD) and (movementLRmean > MEDIUM_MOVING_LR or movementUDmean > MEDIUM_MOVING_UD):
        if movementUDmean < LOW_MOVING_UD:
            cv2.putText(vectorField, 'rotating', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
        else:
            cv2.putText(vectorField, 'looking', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
    else:
        cv2.putText(vectorField, 'stopped', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
    cv2.waitKey(200)

def update_windows(frame, vectorField, plot):
    # display to screen
    cv2.imshow(VIDEO_WINDOW,frame)
    cv2.imshow(VECTOR_WINDOW, vectorField)
    cv2.imshow(PLOT_WINDOW, plot)

def find_features(framequeue, feature_params):
    features = None
    iterations = 1
    while (features is None or len(features) <= MIN_FEATURES):
        old_frame = framequeue[-iterations]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        iterations += 1
        if iterations > len(framequeue):
            raise NoFeatures("Could not find sufficient features in last {0} frames.".format(iterations - 1))
    return (old_gray, features, iterations - 1)


def recalculate_new_features(framequeue, feature_params, lk_params, frame_gray, p0, p1, good_new, good_old, old_gray):
    #old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    #p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #when redrawing throw away old features
    #p1 = p0
    #newFeatures = True
    if not p1 is None and len(p1) > MIN_FEATURES:
        good_new = p1[st==1]
        good_old = p0[st==1]
    else:
        good_new = []
        good_old = []
        p0 = None
        p1 = None
        raise NoFeatures("Did not detect enough features in current frame.")
    return (p0, p1, good_new, good_old)


# statics
# CROPPING
LINES_FROM_TOP = 50
LINES_FROM_BOTTOM = 0

# length of buffer for smoothing
BUFFER_LENGTH = 10
FRAME_QUEUE_LENGTH = 10

# thresholds for navigation vs stopping
# variance thresholds
LOW_VARIANCE_THRESHOLD    = 1
MEDIUM_VARIANCE_THRESHOLD = 5
HIGH_VARIANCE_THRESHOLD   = 10

# movement thresholds
LOW_MOVING_THRESHOLD    = 1
MEDIUM_MOVING_THRESHOLD = 5
HIGH_MOVING_THRESHOLD   = 10

# number of corners to detect
MAX_FEATURES = 100
MIN_FEATURES = 20
RECALC_PERCENTAGE = 0.6

# option to display images to screen
DISPLAY = True

# window names
VIDEO_WINDOW = 'img'
VECTOR_WINDOW = 'vectorField'
PLOT_WINDOW = 'plot'

invideofile = "./IMG_0067.MOV"
#invideofile = "./subj8.avi"
outdir = "./"

filename, ext = os.path.splitext(os.path.basename(invideofile))
outvideofile = os.path.join(os.path.dirname(invideofile), filename + "_processed.mov")

cap = cv2.VideoCapture(invideofile)

cv_fourcc_code, FRAME_RATE, FRAME_HEIGHT, FRAME_WIDTH, VIDEO_LENGTH = extract_capture_metadata(cap)

# make thresholds relative to frame dimensions
LOW_VARIANCE_LR    = LOW_VARIANCE_THRESHOLD / FRAME_WIDTH
MEDIUM_VARIANCE_LR = MEDIUM_VARIANCE_THRESHOLD / FRAME_WIDTH
HIGH_VARIANCE_LR   = HIGH_VARIANCE_THRESHOLD / FRAME_WIDTH

LOW_VARIANCE_UD    = LOW_VARIANCE_THRESHOLD / FRAME_HEIGHT
MEDIUM_VARIANCE_UD = MEDIUM_VARIANCE_THRESHOLD / FRAME_HEIGHT
HIGH_VARIANCE_UD   = HIGH_VARIANCE_THRESHOLD / FRAME_HEIGHT

LOW_MOVING_LR    = LOW_MOVING_THRESHOLD / FRAME_WIDTH
MEDIUM_MOVING_LR = MEDIUM_MOVING_THRESHOLD / FRAME_WIDTH
HIGH_MOVING_LR   = HIGH_MOVING_THRESHOLD / FRAME_WIDTH

LOW_MOVING_UD    = LOW_MOVING_THRESHOLD / FRAME_HEIGHT
MEDIUM_MOVING_UD = MEDIUM_MOVING_THRESHOLD / FRAME_HEIGHT
HIGH_MOVING_UD   = HIGH_MOVING_THRESHOLD / FRAME_HEIGHT

print("LR Variance Low: {0}, Med: {1}, High: {2}".format(LOW_VARIANCE_LR, MEDIUM_VARIANCE_LR, HIGH_VARIANCE_LR))
print("UD Variance Low: {0}, Med: {1}, High: {2}".format(LOW_VARIANCE_UD, MEDIUM_VARIANCE_UD, HIGH_VARIANCE_UD))
print("LR Moving   Low: {0}, Med: {1}, High: {2}".format(LOW_MOVING_LR, MEDIUM_MOVING_LR, HIGH_MOVING_LR))
print("UD Moving   Low: {0}, Med: {1}, High: {2}".format(LOW_MOVING_UD, MEDIUM_MOVING_UD, HIGH_MOVING_UD))

#mp4v works pretty well on wide variety of installs of opencv and ffmpeg
#mpeg-4 video
fourcc = cv2.cv.CV_FOURCC(*'mp4v')

print(fourcc, cv_fourcc_code, FRAME_HEIGHT, FRAME_WIDTH, FRAME_RATE)
writer = cv2.VideoWriter(outvideofile, fourcc, FRAME_RATE, (int(FRAME_WIDTH), int(FRAME_HEIGHT)), True)

# try is to allow finally statement at end to always close files properly
try:
    # create windows if display is selected
    if DISPLAY:
        setup_display_windows()

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = MAX_FEATURES,
                         qualityLevel = 0.3,
                         minDistance = 7,
                         blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_frame = old_frame[LINES_FROM_TOP:, 25:-24]
    #cropmask = cv2.CreateImage(cv2.GetSize(old_frame), cv2.CV_8UC1, 1)
    #cropmask[:LINES_FROM_TOP] = 0

    cropmask = np.copy(old_frame)
    cropmask[:LINES_FROM_TOP] = 0
    cropmask[LINES_FROM_TOP:] = 1

    # CROP FRAME
    #old_frame = old_frame[LINES_FROM_TOP:-LINES_FROM_BOTTOM]

    framequeue = deque(maxlen=FRAME_QUEUE_LENGTH)
    framequeue.append(old_frame)

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    #frameskip
    frameSkip = 0

    #setup histogram
    histogram = np.zeros(MAX_FEATURES, dtype = np.float32)

    #setup variance and movement buffer
    #contains data extracted from previous frames
    varianceLRBuffer = deque(maxlen=BUFFER_LENGTH)
    movementLRBuffer = deque(maxlen=BUFFER_LENGTH)
    varianceUDBuffer = deque(maxlen=BUFFER_LENGTH)
    movementUDBuffer = deque(maxlen=BUFFER_LENGTH)

    #for rendering plot
    plot = np.zeros_like(old_frame)

    while(1):
        frame_count = cap.get(1)
        timestamp = timedelta(seconds=(cap.get(0) / 1000))
        sys.stdout.write("Processed {0} of {1}\r".format(timestamp, VIDEO_LENGTH))
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

        # CROP FRAME
        #frame = frame[LINES_FROM_TOP:-LINES_FROM_BOTTOM]

        if ret==True:
            try:
                iterations = 1
                frame = frame[LINES_FROM_TOP:, 25:-24]
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if p0 is None:
                    p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
                    varianceLRBuffer.append(np.nan)
                    movementLRBuffer.append(np.nan)
                    varianceUDBuffer.append(np.nan)
                    movementUDBuffer.append(np.nan)
                    old_gray = frame_gray.copy()
                    old_frame = frame.copy()
                    framequeue.append(old_frame)
                    cv2.putText(vectorField, 'recalculating', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
                    writer.write(vectorField)
                    update_windows(frame, vectorField, plot)
                    continue
                else:
                    # calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Select good points
                if not p1 is None:
                    good_new = p1[st==1]
                else:
                    good_new = []

                if len(good_new) < MIN_FEATURES or frame_count % 10 == 0:
                    try:
                        old_gray, p0, iterations = find_features(framequeue, feature_params)
                        print("Found {0} features in {1} iteration(s)".format(len(p0), iterations + 1))
                        p0, p1, good_new, good_old = recalculate_new_features(framequeue, feature_params, lk_params, frame_gray, p0, p1, good_new, good_old, old_gray)
                    except Exception as e:
                        print e
                        varianceLRBuffer.append(np.nan)
                        movementLRBuffer.append(np.nan)
                        varianceUDBuffer.append(np.nan)
                        movementUDBuffer.append(np.nan)
                        old_gray = frame_gray.copy()
                        old_frame = frame.copy()
                        framequeue.append(old_frame)
                        cv2.putText(vectorField, 'recalculating', (0,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,225,0))
                        writer.write(vectorField)
                        update_windows(frame, vectorField, plot)
                        continue

                magnitudesLeftRight = []
                magnitudesUpDown    = []

                #store vector magnitudes
                #magnitudes = []

                good_old = p0[st==1]

                for i,(new,old) in enumerate(zip(good_new, good_old)):
                    a,b = new.ravel()
                    c,d = old.ravel()

                    #draw vector field
                    cv2.circle(vectorField,(c,d),1,(0,255,0),-1)
                    cv2.line(vectorField, (a,b), (c,d), (0,255,0), 1)
                    cv2.circle(vectorField, (a,b), 1, (0, 0, 255), -1)

                    #add data to calculate variance
                    #mag = np.sqrt((a-c) ** 2 + (b-d) ** 2)
                    #magnitudes.append(mag)

                    #calculate direction of feature movement
                    magnitudesLeftRight.append(a - c)
                    magnitudesUpDown.append(b -d)

                #variance = np.var(magnitudes) / iterations
                movementLeftRight = np.mean(magnitudesLeftRight) / FRAME_WIDTH / iterations
                movementUpDown    = np.mean(magnitudesUpDown) / FRAME_HEIGHT / iterations
                varianceLeftRight = np.var(magnitudesLeftRight) / FRAME_WIDTH / iterations
                varianceUpDown    = np.var(magnitudesUpDown) / FRAME_WIDTH / iterations

                # append values to buffers
                for i in range(iterations):
                    if i == 0:
                        pass
                    else:
                        varianceLRBuffer.pop()
                        movementLRBuffer.pop()
                        varianceUDBuffer.pop()
                        movementUDBuffer.pop()
                    varianceLRBuffer.append(varianceLeftRight)
                    movementLRBuffer.append(movementLeftRight)
                    varianceUDBuffer.append(varianceUpDown)
                    movementUDBuffer.append(movementUpDown)

                movementLeftRight_mean = np.nanmean(movementLRBuffer)
                movementUpDown_mean    = np.nanmean(movementUDBuffer)
                varianceLeftRight_mean = np.nanmean(varianceLRBuffer)
                varianceUpDown_mean    = np.nanmean(varianceUDBuffer)

                draw_text_overlay(vectorField,
                                  varianceLeftRight_mean, varianceUpDown_mean, movementLeftRight_mean, movementUpDown_mean,
                                  varianceLeftRight, movementLeftRight, varianceUpDown, movementUpDown)

                for index, eachVariance in enumerate(varianceLRBuffer):
                    if not eachVariance is np.nan:
                        cv2.circle(plot, (index, int(eachVariance)), 1, (50,50,255), -1)

                for index, eachVariance in enumerate(varianceUDBuffer):
                    if not eachVariance is np.nan:
                        cv2.circle(plot, (index, int(eachVariance)), 1, (50,255,50), -1)

                # write frame to output
                writer.write(vectorField)

                if DISPLAY:
                    update_windows(frame, vectorField, plot)

            except NoFeatures as e:
                print e
            except Exception as e:
                import traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print e
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)

            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            old_frame = frame.copy()
            framequeue.append(old_frame)

            try:
                p0 = good_new.reshape(-1,1,2)
            except:
                p0 = None


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
