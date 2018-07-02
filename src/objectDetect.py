import cv2

#detect all images in /img folder draws a rectangle on target image
def detect():
    count = 0
    while(True):
        image = cv2.imread("./img/obj-"+str(count)+".jpg")

        height = image.shape[0]
        width = image.shape[1]

        if(image is None):
            break

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
        _,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)
        #threshold
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        dilated = cv2.dilate(thresh,kernel,iterations = 13) # dilate
        _, contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # get contours
        # for each contour found, draw a rectangle around it on original
        for contour in contours:
            # get rectangle bounding contour

            (x,y,w,h) = cv2.boundingRect(contour)
            # (x, y, w, h)
            # discard areas that are too large

            if h>(height*.90) and w>(width*.90):

                continue
            # discard areas that are too small

            if h<(height*0.10) or w<(width*.10):

                continue
            # draw rectangle around contour on original image

            cv2.rectangle(
                image,(x,y),
                (x+w,y+h),
                (255,0,255),2)
        # write original image with added contours to disk

        cv2.imwrite("./img/object_highlighted/obj-"+str(count)+".jpg", image)
        count += 1
