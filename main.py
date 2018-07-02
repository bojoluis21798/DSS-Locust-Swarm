from src.camera import startCam
from src.objectDetect import detect

def main():
    # Start camera.
    startCam()
    # Draw Rectangle around object in the image.
    detect()

if __name__ == "__main__":
	main()
