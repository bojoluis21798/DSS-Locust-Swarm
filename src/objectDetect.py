import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev

# def watershed(img):
#     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

#     kernel = np.ones((3,3),np.uint8)
#     opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

#     sure_bg = cv.dilate(opening,kernel,iterations=3)

#     dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
#     ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

#     sure_fg = np.uint8(sure_fg)
#     unknown = cv.subtract(sure_bg,sure_fg)

#     # Marker labelling
#     ret, markers = cv.connectedComponents(sure_fg)
#     # Add one to all labels so that sure background is not 0, but 1
#     markers = markers+1
#     # Now, mark the region of unknown with zero
#     markers[unknown==255] = 0

#     markers = cv.watershed(img,markers)
#     for row in range(int(len(markers)*0.01), int(len(markers)-(len(markers)*0.01))):
#         for col in range(int(len(markers[row])*0.01), int(len(markers[row])-len(markers[row])*0.01)):
#             if(markers[row][col] == -1):
#                 img[row][col] = [20,255,57]

#     return img

# def contoursWithCanny(img):
#     img = img.copy()
#     edged = cv.Canny(img, 10, 250)

#     #applying closing function
#     kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
#     closed = cv.morphologyEx(edged, cv.MORPH_CLOSE, kernel)
#     #finding_contours
#     _, cnts, _ = cv.findContours(closed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     cnts.sort(key=lambda x: cv.contourArea(x))
#     #cnts = cnts[:2]

#     tooSmall = img.size * 0.05
#     for c in cnts:
#         # if(cv.contourArea(c) < tooSmall):
#         #     continue

#         x,y,w,h = cv.boundingRect(c)
#         # if w>50 and h>50:
#         #     cropped=img[y:y+h,x:x+w]
#         #     return cropped

#         cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

#         cv.drawContours(img, c, -1, (0,255,0), 2)
#     return img

# def contoursWithStaticSaliency(img):
#     img = contoursWithSobel(img)
#     noise = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)

#     saliency = cv.saliency.StaticSaliencyFineGrained_create()
#     (success, saliencyMap) = saliency.computeSaliency(img)
#     threshMap = cv.threshold(saliencyMap, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

#     kernel = np.ones((30, 30), np.uint8)
#     closing = cv.morphologyEx(threshMap, cv.MORPH_CLOSE, kernel)

#     # edged = cv.Canny(closing, 0, 250)

#     _, cnts, heirarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#     tooSmall = img.size * 0.03
#     for c in cnts:
#         # if cv.contourArea(c) < tooSmall:
#         #     continue

#         rect = cv.minAreaRect(c)
#         if rect[1][0]*rect[1][1] > tooSmall:
#             box = cv.boxPoints(rect)
#             box = np.int0(box)
#             img = cv.drawContours(img,[box],0,(0,255,0),5)

#     return img

# def segmentApproach(img):
#     return contoursWithStaticSaliency(img)

def contoursWithSobel(img):
    def edgedetect (channel):
        sobelX = cv.Sobel(channel, cv.CV_16S, 1, 0)
        sobelY = cv.Sobel(channel, cv.CV_16S, 0, 1)
        sobel = np.hypot(sobelX, sobelY)

        sobel[sobel > 255] = 255; # Some values seem to go above 255. However RGB channels has to be within 0-255
        return sobel


    def findSignificantContours (img, edgeImg):
        image, contours, heirarchy = cv.findContours(edgeImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Find level 1 contours
        level1 = []
        for i, tupl in enumerate(heirarchy[0]):
            # Each array is in format (Next, Prev, First child, Parent)
            # Filter the ones without parent
            if tupl[3] == -1:
                tupl = np.insert(tupl, 0, [i])
                level1.append(tupl)

        significant = []
        tooSmall = edgeImg.size * 5 / 100
        height, width, _ = img.shape
        min_x, min_y = width, height
        max_x = max_y = 0

        for tupl in level1:
            contour = contours[tupl[0]]
            contour = cv.convexHull(contour)
            area = cv.contourArea(contour)
            if area > tooSmall:
                significant.append([contour, area])

                (x,y,w,h) = cv.boundingRect(contour)
                min_x, max_x = min(x, min_x), max(x+w, max_x)
                min_y, max_y = min(y, min_y), max(y+h, max_y)

                # Draw the contour on the original image
                cv.drawContours(img, [contour], 0, (0, 255, 0), 2, cv.LINE_AA, maxLevel=1)

        significant.sort(key=lambda x: x[1])
        # print ([x[1] for x in significant])
        return [x[0] for x in significant]


    blurred = cv.GaussianBlur(img,(5,5),0)
    edgeImg = np.max( np.array([ edgedetect(blurred[:,:, 0]), edgedetect(blurred[:,:, 1]), edgedetect(blurred[:,:, 2]) ]), axis=0 )
    mean = np.mean(edgeImg)
    edgeImg[edgeImg <= mean] = 0
    edgeImg[edgeImg > 255] = 255

    edgeImg_8u = np.asarray(edgeImg, np.uint8)
    # Find contours and bounding box coordinates for bounding all contours
    significant, min_x, min_y, max_x, max_y = findSignificantContours(img, edgeImg_8u)

    # Mask
    mask = edgeImg.copy()
    mask[mask > 0] = 0
    cv.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)

    #Finally remove the background
    img[mask] = 0
    return img, significant

# def smoothingInterpolate(img_contour, contours, orig_img):

#     smoothened = []

#     for contour in contours:
#         x,y = contour.T
#     # Convert from numpy arrays to normal arrays
#         x = x.tolist()[0]
#         y = y.tolist()[0]

#         tck, u = splprep([x,y], u=None, s=1.0, per=1)
#         u_new = np.linspace(u.min(), u.max(), 25)
#         x_new, y_new = splev(u_new, tck, der=0)
#         res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
#         smoothened.append(np.asarray(res_array, dtype=np.int32))

#     # Overlay the smoothed contours on the original image
#     cv.drawContours(orig_img, smoothened, -1, (255,255,255), 2)

#     return orig_img

#for rectangular shapes only, not applicable for shapes with curves
def smoothingPoly(img_contour, contours, orig_img):
    index = 0

    while(index < len(contours)):
        epsilon = 0.01*cv.arcLength(contours[index],True)
        approx = cv.approxPolyDP(contours[index], epsilon,True)
        contours[index] = approx
        index+=1

    cv.drawContours(orig_img, contours, -1, (255,255,255), 2)
    return orig_img, contours

# def orderingContours(img, contours):
#     (contours,_) = cnts.sort_contours(contours)
#     colors = ((0, 0, 255), (240, 0, 159), (255, 0, 0), (255, 255, 0))

#     for(i, c) in enumerate(contours):
#         box = cv.minAreaRect(c)
#         box = cv.boxPoints(box)
#         box = np.array(box,dtype = "int")
#         cv.drawContours(img,[box], -1, (0,255,0), 2)
#         rect = perspective.order_points(box)

#         for ((x, y), color) in zip(rect, colors):
# 		    cv.circle(img, (int(x), int(y)), 5, color, -1)

#         # draw the object num at the top-left corner
#         cv.putText(img, "Object #{}".format(i + 1), (int(rect[0][0] - 15), int(rect[0][1] - 15)), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

#     return img

def sobelApproach(img):
    image_with_contour, contours = contoursWithSobel(img)
    smoothened_contour_img, smooth_contours = smoothingPoly(image_with_contour, contours, img)

    return smoothened_contour_img, smooth_contours
    # return getScale(img, smooth_contours)

def detectAll():
    i = 0
    images = []
    contours = []
    while(True):
        img = cv.imread("./img/obj-"+str(i)+".jpg")

        if img is None:
            break
        print "Detecting obj-"+str(i)+".jpg ..."
        imgWithContours, smoothContours = sobelApproach(img)
        cv.imwrite("./img/object_highlighted/obj-"+str(i)+".jpg", imgWithContours)

        i += 1
        images.append(img)
        contours.append(smoothContours)
    return images, contours


def detect(imgId):
    img = cv.imread("./img/obj-"+str(imgId)+".jpg")
    print "Detection obj-"+str(imgId)+".jpg ..."
    imgWithContours, smooth_contours = sobelApproach(img)
    cv.imwrite("./img/object_highlighted/obj-"+str(imgId)+".jpg", imgWithContours)
    return img, smooth_contours
