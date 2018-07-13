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
    img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    # filter black color
    mask1 = cv.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 125]))
    mask1 = cv.morphologyEx(mask1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
    mask1 = cv.Canny(mask1, 100, 300)
    mask1 = cv.GaussianBlur(mask1, (1, 1), 0)
    mask1 = cv.Canny(mask1, 100, 300)

    # mask1 = cv.morphologyEx(mask1, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)))

    # Find contours for detected portion of the image
    im2, cnts, hierarchy = cv.findContours(mask1.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:2] # get largest five contour area
    rects = []
    for c in cnts:
        #peri = cv.arcLength(c, True)
        # approx = cv.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv.boundingRect(c)
        if h >= 15:
            # if height is enough
            # create rectangle for bounding
            rect = (x, y, w, h)
            rects.append(rect)
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1);

    # for row in range(len(cnts)):
    #     for col in range(len(cnts[row])):
    #         x = cnts[row][col][0][0]
    #         y = cnts[row][col][0][1]

    #         img[x][y] = [20,255,57]

    return mask1

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
