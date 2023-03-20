import cv2
import numpy as np



def threshHolding(image):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ## URBAN
    LowerWhite = np.array([0,179,0])
    UpperWhite = np.array([255,186,255])
    maskWhite = cv2.inRange(imgHSV, LowerWhite, UpperWhite)
    maskWhite = cv2.cvtColor(maskWhite, cv2.COLOR_GRAY2RGB)
    return maskWhite


def hsv_window():
    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 640, 240)
    cv2.createTrackbar("HUE Min", "HSV", 0, 179, lambda x:None)
    cv2.createTrackbar("HUE Max", "HSV", 179, 179, lambda x:None)
    cv2.createTrackbar("SAT Min", "HSV", 0, 255, lambda x:None)
    cv2.createTrackbar("SAT Max", "HSV", 255, 255, lambda x:None)
    cv2.createTrackbar("VALUE Min", "HSV", 0, 255, lambda x:None)
    cv2.createTrackbar("VALUE Max", "HSV", 255, 255, lambda x:None)


def hsv_trackbar(img):
    img = cv2.resize(img, (256,256))
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
 
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
 
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img,result])
    # cv2.imshow('Horizontal Stacking', hStack) 
    return result






def initializeTrackbars(intialTracbarVals,wT=256, hT=256):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2,  lambda x:None)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT,  lambda x:None)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2,  lambda x:None)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT,  lambda x:None)    

def valTrackbars(wT=256, hT=256):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points



def warpImage(image, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    

    

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(image,matrix,(w,h))
    return imgWarp


def drawPoints(image, points):
    # print(points)
    for x in range(4):
        cv2.circle(image,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)   
    return image




def getHistogram(image):
    

    histValues = np.sum(image, axis=1)
    print(histValues.shape)