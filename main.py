from src.camera import startCam
from src.objectDetect import detect
import sys

def main():
    if(len(sys.argv) == 1):
        # Start camera.
        startCam()
        # Draw Rectangle around object in the image.
        detect()
    elif(sys.argv[1] == "skipcam"):
        # proceed to detection.
        detect()
    else:
        print "No such command"

if __name__ == "__main__":
	main()
