from src.camera import startCam
from src.objectDetect import detect
import sys

def main():
    if(len(sys.argv) == 1):
        # Start camera.
        startCam()
        # Draw Rectangle around object in the image.
        detect()
    elif(len(sys.argv) == 2 and sys.argv[1] == "skipcam"):
        # proceed to detection.
        detect()
    elif(len(sys.argv) == 3 and sys.argv[1] == "test"):
        detect(int(sys.argv[2]))
    else:
        print "No such command"

if __name__ == "__main__":
	main()
