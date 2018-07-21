from src.camera import startCam
from src.objectDetect import detect, detectAll
from src.scaling import getScale
import sys

def main():
    if(len(sys.argv) == 1):
        # Start camera.
        startCam()
        # Draw Rectangle around object in the image.
        imgs, contours = detectAll()

        scaledImgs = []
        i = 0
        for img, c in zip(imgs, contours):
            scaledImg = getScale(i, img, c)
            scaledImgs.append(scaledImg)
            i += 1
    elif(len(sys.argv) == 2 and sys.argv[1] == "skipcam"):
        # proceed to detection.
        imgs, contours = detectAll()

        scaledImgs = []
        i = 0
        for img, c in zip(imgs, contours):
            scaledImg = getScale(i, img, c)
            scaledImgs.append(scaledImg)
            i += 1
    elif(len(sys.argv) == 3 and sys.argv[1] == "test"):
        imgId = int(sys.argv[2])
        img, contours = detect(imgId)
        scales = getScale(imgId, img, contours)
    else:
        print "No such command"

if __name__ == "__main__":
	main()
