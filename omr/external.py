import cv2
import numpy as np

def rectContour(contours):

    rectCon = []
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            # print("approximate value is : ",len(approx))
            if len(approx) == 4: # get 4 dot for that rectangle
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea,reverse=True)
    # print("in rectcontour array : ",len(rectCon))
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def splitBoxes(img):
    rows = np.vsplit(img,25)
    # cv2.imshow('split',rows[0])
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,4)
        for box in cols:
            boxes.append(box)
            # cv2.imshow('box',box)
            # cv2.imshow('')
    return boxes


def showAnswers(img,testIndex,score,questions=25,choices=4):
    secW = int(img.shape[1]/choices)
    secH = int(img.shape[0]/questions)

    for x in range(0,questions):
        myAns= testIndex[x]
        cX = (myAns * secW) + secW // 2
        cY = (x * secH) + secH // 2
        myscore = score[x]
        if myscore==1:
            # print(type(grading[x]))
            myColor1 = (0,255,0)
            axesLength = (20, 10) 
            angle,startAngle,endAngle = 0,0,360
            cv2.ellipse(img,(cX,cY),axesLength,angle,startAngle,endAngle,myColor1,cv2.FILLED)
    
        else:
            myColor2 = (255,0,0)
            myColor1 = (0,255,0)
            axesLength = (20, 10) 
            angle,startAngle,endAngle = 0,0,360
            cv2.ellipse(img, (cX, cY),axesLength,angle,startAngle,endAngle, myColor2, cv2.FILLED)
    return img


def reorder(myPoints):

    myPoints = myPoints.reshape((4, 2)) # REMOVE EXTRA BRACKET
    # print(myPoints)
    myPointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = myPoints.sum(1)
    # print(add)
    # print(np.argmax(add))
    myPointsNew[0] = myPoints[np.argmin(add)]  #[0,0]
    myPointsNew[3] =myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]  #[w,0]
    myPointsNew[2] = myPoints[np.argmax(diff)] #[h,0]

    return myPointsNew