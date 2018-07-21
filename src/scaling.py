from imutils import contours as cnts
from imutils import perspective
import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

#same with orderingContours() but now measures the contours
def getScale(id, img, contours):
    (contours,_) = cnts.sort_contours(contours)
    pixelsPerInch = None

    for c in contours:
        box = cv.minAreaRect(c)
        box = cv.boxPoints(box)
        box = np.array(box,dtype = "int")

        box = perspective.order_points(box)
        cv.drawContours(img,[box.astype("int")], -1, (0,255,0), 2)

        for (x, y) in box:
            cv.circle(img, (int(x), int(y)), 5, (0,0,255), -1)

        (tl,bl,br,tr) = box
        (tltrX , tltrY) = midpoint(tl,tr)
        (blbrX, blbrY) = midpoint(bl,br)

        (tlblX,tlblY) = midpoint(tl,bl)
        (trbrX, trbrY) = midpoint(tr,br)

        cv.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

        # draw lines between the midpoints
        cv.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        if(pixelsPerInch is None):
            pixelsPerInch = dA / 12.0

        # compute the size of the object
        dimA = dA / pixelsPerInch
        dimB = dB / pixelsPerInch

        # draw the object sizes on the image
        cv.putText(img, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)),
            cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv.putText(img, "{:.1f}in".format(dimB),
            (int(trbrX + 10), int(trbrY)), cv.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2)

        cv.imwrite("./img/object_scales/obj-"+str(id)+".jpg", img)

    return img
