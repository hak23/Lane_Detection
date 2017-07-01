import sys
import numpy as np
import cv2

'''
:Author Vignesh Ungrapalli
A basic lane detection program with line detection in Hough Space and a simple line fitting.

Important note: The code for carving out unnecessary region (in preprocess_image) is very much
camera placement dependent. I have tried the code on videos where everything goes haywire because
the region carving completely ignores the lanes. I still think in most cases removing the upper sky part should
work.
'''

VIDEO_INPUT = 'solidWhiteRight.mp4'
# VIDEO_INPUT = 'challenge.mp4'

RHO_ACCURACY    = 1
THETA_ACCURACY  = np.pi/90 
MIN_VOTES       = 20
MIN_LINE_LENGTH = 20
MAX_LINE_GAP    = 5

FRAME_DELAY = 25

YELLOW_LOW  = np.asarray([20, 100, 100]) 
YELLOW_HIGH = np.asarray([30, 255, 255])
WHITE_LOW   = np.asarray([0, 0, 230])
WHITE_HIGH  = np.asarray([255, 80, 255])


def get_slope(x1, y1, x2, y2):
    '''
    Get slope of line
    
    :param x1, y1, x2, y2: 4 scalars with (x1,y1) and (x2,y2) as points

    :ret dy/dx : slope of the line represented by thr two points
    :ret None: In case there is a divide by zero
    '''

    if x2 != x1:
        dy = float(y2) - float(y1)
        dx = float(x2) - float(x1)

        return dy/dx
    else:
        return None


def get_line_props(line):
    '''
    Get the slope and intercept of the line defined by collinear vector and a point on the line

    :param line: A list of four elements [x1,y1,x2,y2] where (x1,y1) is a vector colliniear to the given line
                 and (x2,y2) is a point on the line. Basically output of cv2.fitLine

    :ret m,c: slope and intercept of the line passing through (x1,y1) and (x2,y2)
    '''

    if line[0] is not 0:
        m = (float(line[1])/line[0])
        c = float(line[3]) - (m * line[2])
    else:
        m = None
        c = None

    return m,c


def get_x(y, lane):
    '''
    Get the x coordinate give line and y coordinate.
   
    :param y: y coordinate of the point whose x coordinate needs to be calculated
    :param lane: A list of four elements [x1,y1,x2,y2] where (x1,y1) is a vector colliniear to the given line
                 and (x2,y2) is a point on the line. Basically output of cv2.fitLine

    :ret: x coordinate of the point
    :ret None: In case the slope is zero. It is not a problem as we do not consider lines of slope zero for our lanes.
    '''
    
    m, c = get_line_props(lane)

    if m is not 0:
        return ((y-c)/m)
    else:
        return None


def color_threshold(frame_in, frame_hsv):
    '''
    Color thresholding on the input image. Black out the pixels which do not fall in the 
    range of yellow and white thresholds

    :param frame_in: Input frame
    :param frame_hsv: Input frame in hsv space

    :ret threshold: The thresholded frame 
    '''

    yellow_mask = cv2.inRange(frame_hsv, YELLOW_LOW, YELLOW_HIGH)
    white_mask  = cv2.inRange(frame_hsv, WHITE_LOW, WHITE_HIGH)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    threshold = cv2.bitwise_and(frame_in, frame_in, mask=mask)
    # cv2.imshow('',threshold)

    return threshold


def preprocess_image(frame_in):
    '''
    Do some preprocessing on the input frame. Discard not needed colors (Lane lines are mostly white or yellow).
    Get the edge image and carve out unnecessary region in the frame. Note that the carve out region depends on
    the placement of camera.

    :param frame_in: Input frame

    :ret frame_edge: preprocessed frame
    '''

    frame_h = frame_in.shape[0] 
    frame_w = frame_in.shape[1]
    
    frame_hsv = cv2.cvtColor(frame_in, cv2.COLOR_BGR2HSV)
    frame_thresholded = color_threshold(frame_in, frame_hsv) 
    frame_hsv = cv2.split(frame_thresholded)
    frame_edge = cv2.Canny(frame_hsv[2], 50, 150, apertureSize=3)
    
    # Carve out unnecessary region in the frame
    pts = np.array([[(0,0), (0, frame_h), (frame_w/2 - 25, frame_h/2 + 25), (frame_w/2 + 25, frame_h/2 + 25), (frame_w, frame_h), (frame_w, 0)]])
    cv2.fillPoly(frame_edge, pts, 0)
    cv2.imshow('w', frame_edge) 
    
    return frame_edge


def get_line_list(frame_in):
    '''
    Use Hough transform to get Lines in terms of point coordinates
    
    :param frame_in: Input frame

    :ret lines_list: A list of Lines (in terms of  point coordinates)
    '''

    lines_list = cv2.HoughLinesP(frame_in, RHO_ACCURACY, THETA_ACCURACY, MIN_VOTES, MIN_LINE_LENGTH, MAX_LINE_GAP)
    # omit the first dimension
    lines_list = np.squeeze(lines_list, 1)

    return lines_list


def process_lines(lines_list):
    '''
    Distinguish points into rigth lane, left lane and neither(ignore)

    :param lines_list: A list of lines represented by two points. Basically the output of cv2.HoughLinesP

    :ret right_lane, left_lane: Two lists. One containing the points forming the right lane and other the left lane
    '''
    
    right_lane = []
    left_lane  = []

    for lines in lines_list:
        slope = get_slope(lines[0], lines[1], lines[2], lines[3])
        # These lines would not be on either lanes.
        if -0.2 < slope < 0.2 :
            continue
        
        # Collect points adding to right lane (potentially)
        if slope > 0:
            right_lane.append((lines[0], lines[1]))
            right_lane.append((lines[2], lines[3]))
        # Collect points adding to the left lane (potentially)
        else:
            left_lane.append((lines[0], lines[1]))
            left_lane.append((lines[2], lines[3]))
   
    return right_lane, left_lane



if __name__ == '__main__':
    ''' 
    Read Video frame by frame and call hepler functions.
    '''
    
    if len(sys.argv) > 1:
        # Expcet Video Filename in argv[1]
        video_input = sys.argv[1]
    else:
        # use default video in case there is no input from console
        video_input = VIDEO_INPUT

    video_in = cv2.VideoCapture(video_input)
    if video_in.isOpened():
        FRAME_WIDTH  = video_in.get(3)
        FRAME_HEIGHT = video_in.get(4)
        print FRAME_HEIGHT, FRAME_WIDTH

    while(video_in.isOpened()):
        # Read the video frame by frame
        ret, frame_in = video_in.read()
        if ret:
            # preprocess input frame
            frame_ = preprocess_image(frame_in)

            # Get list of lines with hough transform on frame 
            lines_list = get_line_list(frame_)
    
            # check if lines_list is not empty
            if np.any(np.equal(lines_list, None)):
                # print "skipping frame"
                continue
            else:
                right_lane, left_lane = process_lines(lines_list)

            # Draw the right lane on the original frame
            if len(right_lane) is not 0:
                right_lane = cv2.fitLine(np.asarray(right_lane), cv2.DIST_L1, 0, 0.01, 0.01)
                
                y1 = FRAME_HEIGHT
                x1 = get_x(y1, right_lane)
                y2 = FRAME_HEIGHT * 0.75
                x2 = get_x(y2, right_lane)
                cv2.line(frame_in, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 5)
            
            # Draw the left lane on the original frame
            if len(left_lane) is not 0:
                left_lane  = cv2.fitLine(np.asarray(left_lane), cv2.DIST_L1, 0, 0.01, 0.01)
               
                y1 = FRAME_HEIGHT
                x1 = get_x(y1, left_lane)
                y2 = FRAME_HEIGHT * 0.75
                x2 = get_x(y2, left_lane)
                cv2.line(frame_in, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 5)

            cv2.imshow('output', frame_in)

            if cv2.waitKey(FRAME_DELAY) & 0xFF == ord('q'):
                break 
        else:
            print "Nothing to read"
            break

    # When everything done, release the capture
    video_in.release()
    cv2.destroyAllWindows()

