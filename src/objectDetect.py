import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def watershed(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

    sure_bg = cv.dilate(opening,kernel,iterations=3)

    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    for row in range(int(len(markers)*0.01), int(len(markers)-(len(markers)*0.01))):
        for col in range(int(len(markers[row])*0.01), int(len(markers[row])-len(markers[row])*0.01)):
            if(markers[row][col] == -1):
                img[row][col] = [20,255,57]

    return img

def contours(img):
    img = img.copy()
    edged = cv.Canny(img, 10, 250)

    #applying closing function
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
    #finding_contours
    _, cnts, _ = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # for c in cnts:
    #     peri = cv.arcLength(c, True)
    #     approx = cv.approxPolyDP(c, 0.02 * peri, True)
    #     cv.drawContours(img, [approx], -1, (0, 255, 0), 2)

    for c in cnts:
        cv.drawContours(img, c, -1, (0, 255, 0), 5)

    return img

def detectAll():
    i = 0
    while(True):
        img = cv.imread("./img/obj-"+str(i)+".jpg")

        if img is None:
            break

        ret = watershed(img)
        cv.imwrite("./img/object_highlighted/obj-"+str(i)+".jpg", ret)
        i += 1

def detect(imgId):
    img = cv.imread("./img/obj-"+str(imgId)+".jpg")
    ret = contours(img)
    cv.imwrite("./img/object_highlighted/obj-"+str(imgId)+".jpg", ret)
