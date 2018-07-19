import cv2

count = 0

def __saveFrame(frame):
	global count
	cv2.imwrite('./img/obj-'+str(count)+'.jpg', frame)
	count+= 1

def startCam():
	cap = cv2.VideoCapture(0)

	while(True):
		ret, frame = cap.read()

		cv2.imshow('DSS', frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		elif cv2.waitKey(1) & 0xFF == ord('e'):
			__saveFrame(frame)

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
