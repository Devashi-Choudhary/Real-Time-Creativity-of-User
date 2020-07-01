import argparse
import numpy as np
import cv2
import operator
import matplotlib.pyplot as plt
from collections import deque
from scipy.spatial import distance
from sklearn.metrics import pairwise
from keras.models import model_from_json
from keras import models

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default = "mods.h5", help = "path to model")
args = vars(ap.parse_args())

classifier = models.load_model(args['model'])
class_dict = {0:'The-Eiffel-Tower', 1:'Apple', 2:'Cup', 3:'Laptop', 4:'Leaf', 5:'Penguin', 6:'Pizza', 7:'Triangle', 8:'Shoe', 9:'Wine-Bottle'}

def get_prediction(creative_image, classifier, class_dict):
    
    creative_image = creative_image.astype(np.uint8)
    gray_creative = cv2.cvtColor(creative_image, cv2.COLOR_BGR2GRAY)
    median_blur = cv2.medianBlur(gray_creative, 15)
    
    label = "Nothing"
    thresh_img = cv2.threshold(median_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    creative_cnts = cv2.findContours(thresh_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    if len(creative_cnts) >= 1:
        max_cnt = max(creative_cnts, key=cv2.contourArea)
        
        if cv2.contourArea(max_cnt) > 150:
            x, y, w, h = cv2.boundingRect(max_cnt)
            drawn_image = gray_creative[y:y + h, x:x + w]
            cv2.imshow("cf",drawn_image)
            t = cv2.resize(drawn_image, (28,28))
            t = t.reshape(1,1,28,28)
            prediction = classifier.predict_classes(t)
            label = class_dict[prediction[0]]
    return label


# # Initialization
region_x = 0.5
region_y = 0.6
count_of_frames = 40
alpha = 0.5
threshold = 25
region_x_temp = 0.5
region_y_temp = 0.9

# # Detect Background for Background Subtraction
def detect_background(image, alpha):
    
    global background
    if background is None:
        background = np.copy(image).astype(np.float64)
        return
    
    cv2.accumulateWeighted(image, background, alpha)

# # Find Centroid of the Detected Hand
def find_centroid(cont):
    
    moment = cv2.moments(cont)
    if moment['m00'] != 0:
        cx = int(moment['m10']/moment['m00'])
        cy = int(moment['m01']/moment['m00'])
        return cx, cy
    return None

# # Find the Farthest point from the Centroid
def farthest_point(defects, contour, centroid):
    
    if defects is not None and centroid is not None:
        starting_point = defects[:, 0][:, 0]
        cx, cy = centroid
        
        x = np.array(contour[starting_point][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[starting_point][:, 0][:, 1], dtype=np.float)

        xp = (x-cx)**2
        yp = (y-cy)**2
        dist = cv2.sqrt(xp+yp)

        max_dist_index = np.argmax(dist)

        if max_dist_index < len(starting_point):
            farthest_defect = starting_point[max_dist_index]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None

# # Count the number of Fingers
def count_fingers(threshold_image, max_contour):
    
    convex = cv2.convexHull(max_contour)
    top    = tuple(convex[convex[:, :, 1].argmin()][0])
    bottom = tuple(convex[convex[:, :, 1].argmax()][0])
    left   = tuple(convex[convex[:, :, 0].argmin()][0])
    right  = tuple(convex[convex[:, :, 0].argmax()][0])

    cX = int((left[0] + right[0]) / 2)
    cY = int((top[1] + bottom[1]) / 2)
    
    Xs=[(cX, cY)]
    Ys=[left, right, top, bottom]
    
    dist = pairwise.euclidean_distances(Xs, Y=Ys)[0]
    max_dist = dist[dist.argmax()]
    
    radius = int(0.9 * max_dist)
    circum = (2 * np.pi * radius)
    circle_roi = np.zeros(threshold_image.shape[:2], dtype=np.uint8)
    cv2.circle(circle_roi, (cX, cY), radius, 255, 1)
    
    circle_roi = cv2.bitwise_and(threshold_image, threshold_image, mask=circle_roi)
#     cv2.imshow("circle_roi", circle_roi)
    cnts, _ = cv2.findContours(circle_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        
        if ((cY + (cY * 0.25)) > (y + h)) and ((circum * 0.40) > c.shape[0]):
            count += 1

    return count 

# # Capture Video and Finger Tracking
background = None
fcount = 0
pts = deque(maxlen=512)

cap = cv2.VideoCapture(0)
background = None
start = False
ret, frame = cap.read()
# top_left = (int(region_x*frame.shape[1]), 0)
# bottom_right = (frame.shape[1], int(region_y*frame.shape[0]))
# required_shape = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
save_image = np.zeros(shape=frame.shape)

while True:
    ret, frame = cap.read()

    ''' -----Flip the image so that it is not mirror image----- '''
    frame = cv2.flip(frame, 1)
    
    copy_frame = np.copy(frame)
    top_left = (int(region_x_temp*frame.shape[1]), 0)
    bottom_right = (frame.shape[1], int(region_y_temp*frame.shape[0]))
    
    top_left_copy = (int(region_x*frame.shape[1]), 0)
    bottom_right_copy = (frame.shape[1], int(region_y*frame.shape[0]))
    
    ''' -----Find Region of Interest (ROI)----- '''
    roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    if fcount < count_of_frames:
        detect_background(gray, alpha)
    
    else:
        ''' -----Background Subtraction----- '''
        difference = cv2.absdiff(background.astype(np.uint8), gray)
        ret, threshold_image = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
        
        ''' -----Finding Contours from the Frame----- '''
        contours, hierachy = cv2.findContours(threshold_image ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
#             print("Put your hand in the frame")
            cv2.putText(frame, "Can't detect anything", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,150), 2)
            
        else:
        
            try:
                if start:
                    ''' -----Detecting Hand and draw it on the Frame----- '''
                    max_contour = max(contours, key = cv2.contourArea)
                    area = cv2.contourArea(contours[0])
                    cv2.drawContours(copy_frame, [max_contour + top_left],  -1, (0, 255, 0), 3)
                    cv2.imshow("Thesholded", threshold_image)

    #                 if area > 3000:
                    ''' -----Find Cetroid of the detected Hand----- '''
                    centroid = find_centroid(max_contour)
                    if centroid is not None:
                        actual_centroid = tuple(map(operator.add, centroid, top_left))
                        cv2.circle(copy_frame, actual_centroid, 5, [255, 0, 255], -1)

                    ''' -----Calculate Convex hull and convexity Defects.. Using them calculate farthest point from the centroid----- '''
                    hull = cv2.convexHull(max_contour, returnPoints=False)
                    defects = cv2.convexityDefects(max_contour, hull)
                    far_point = farthest_point(defects, max_contour, centroid)
#                     print("Centroid : " + str(centroid) + ", farthest Point : " + str(far_point))
                    
                    number = count_fingers(threshold_image, max_contour)
                    cv2.putText(copy_frame, str(number), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    if far_point is not None:
                        
                        center = tuple(map(operator.add, far_point, top_left))
                        cv2.circle(copy_frame, center, 5, [0, 0, 255], -1)
                        
                        if number == 1:
                            pts.appendleft(center)
                        else:
                            pts = deque(maxlen=512)

                    ''' -----Draw according to the move of the fingers----- '''
                    for i in range(1, len(pts)):
                        if pts[i - 1] is None or pts[i] is None:
                            continue
                        cv2.line(save_image, pts[i - 1], pts[i], (255, 255, 255), 6)
                        cv2.line(copy_frame, pts[i - 1], pts[i], (0, 0, 255), 2)
                        drawn_image = save_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


                        if len(pts)  != []:
                            print("prediction")
                            predict = get_prediction(save_image, classifier, class_dict)
                            cv2.putText(copy_frame, predict, (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            if(predict != "Nothing"):
                                print(predict)
                                smallImg=cv2.imread('data/'+predict+'.png')
                                img = cv2.resize(smallImg,         (int(copy_frame.shape[1]/4),int(copy_frame.shape[0]/4)),interpolation=cv2.INTER_NEAREST)
                                copy_frame[150:150+img.shape[0], 70:70+img.shape[1]] = img
                            
            except Exception as e:
                print(e)
                cv2.putText(copy_frame, "Put your hand in the frame", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
    ''' -----Draw rectangle to the Region of Interest!----- '''
    x = cv2.rectangle(copy_frame, top_left_copy, bottom_right_copy, (128, 255, 0), 2)
    fcount += 1
    cv2.imshow('drawn Image', save_image/np.max(save_image))
    cv2.imshow('frame', copy_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        start = True
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
