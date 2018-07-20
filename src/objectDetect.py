import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

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

def contoursWithCanny(img):
    img = img.copy()
    edged = cv.Canny(img, 10, 250)

    #applying closing function
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
    #finding_contours
    _, cnts, _ = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cnts.sort(key=lambda x: cv.contourArea(x))
    #cnts = cnts[:2]

    tooSmall = img.size * 0.05
    for c in cnts:
        # if(cv.contourArea(c) < tooSmall):
        #     continue

        x,y,w,h = cv.boundingRect(c)
        # if w>50 and h>50:
        #     cropped=img[y:y+h,x:x+w]
        #     return cropped

        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

        cv.drawContours(img, c, -1, (0,255,0), 2)
    return img

def contoursWithSobel(img):
    def findSignificantContours (img, sobel_8u):
        image, contours, heirarchy = cv.findContours(sobel_8u, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Find level 1 contours
        level1 = []
        for i, tupl in enumerate(heirarchy[0]):
            # Each array is in format (Next, Prev, First child, Parent)
            # Filter the ones without parent
            if tupl[3] == -1:
                tupl = np.insert(tupl, 0, [i])
                level1.append(tupl)

        # From among them, find the contours with large surface area.
        significant = []
        tooSmall = sobel_8u.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
        for tupl in level1:
            contour = contours[tupl[0]];
            area = cv.contourArea(contour)

            if area > tooSmall:
                contour = cv.convexHull(contour)
                cv.drawContours(img, [contour], 0, (0,255,0),2, cv.LINE_AA, maxLevel=1)
                significant.append([contour, area])

        significant.sort(key=lambda x: x[1])
        return [x[0] for x in significant];

    def getSobel (channel):

        sobelx = cv.Sobel(channel, cv.CV_16S, 1, 0, borderType=cv.BORDER_REPLICATE)
        sobely = cv.Sobel(channel, cv.CV_16S, 0, 1, borderType=cv.BORDER_REPLICATE)
        sobel = np.hypot(sobelx, sobely)

        return sobel;

    blurred = cv.GaussianBlur(img, (5, 5), 0) # Remove noise

    # Edge operator
    sobel = np.max( np.array([ getSobel(blurred[:,:, 0]), getSobel(blurred[:,:, 1]), getSobel(blurred[:,:, 2]) ]), axis=0 )

    # Noise reduction trick, from http://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l182
    mean = np.mean(sobel)

    # Zero any values less than mean. This reduces a lot of noise.
    sobel[sobel <= mean] = 0;
    sobel[sobel > 255] = 255;

    cv.imwrite('output/edge.png', sobel);

    sobel_8u = np.asarray(sobel, np.uint8)

    # Find contours
    significant = findSignificantContours(img, sobel_8u)

    # Mask
    mask = sobel.copy()
    mask[mask > 0] = 0
    cv.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)

    #Finally remove the background
    img[mask] = 0;

    return img

def contoursWithStaticSaliency(img):
    img = img.copy()
    noise = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)

    saliency = cv.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    threshMap = cv.threshold(saliencyMap, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    kernel = np.ones((30, 30), np.uint8)
    closing = cv.morphologyEx(threshMap, cv.MORPH_CLOSE, kernel)

    # edged = cv.Canny(closing, 0, 250)

    _, cnts, heirarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    tooSmall = img.size * 0.03
    for c in cnts:
        # if cv.contourArea(c) < tooSmall:
        #     continue
        cv.drawContours(img, [c], -1, (255,0,0), 5)
        rect = cv.minAreaRect(c)
        if rect[1][0]*rect[1][1] > tooSmall:
            box = cv.boxPoints(rect)
            box = np.int0(box)
            img = cv.drawContours(img,[box],0,(0,255,0),5)

    return img

def segmentApproach(img):
    return contoursWithStaticSaliency(img)

def detectAll():
    i = 0
    while(True):
        img = cv.imread("./img/obj-"+str(i)+".jpg")

        if img is None:
            break
        print "Detecting obj-"+str(i)+".jpg ..."
        ret = segmentApproach(img)
        cv.imwrite("./img/object_highlighted/obj-"+str(i)+".jpg", ret)
        i += 1

def detect(imgId):
    img = cv.imread("./img/obj-"+str(imgId)+".jpg")
    print "Detection obj-"+str(imgId)+".jpg ..."
    ret = segmentApproach(img)
    cv.imwrite("./img/object_highlighted/obj-"+str(imgId)+".jpg", ret)
