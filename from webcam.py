import numpy as np
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import time
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor","book"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('/home/kd710306/Documents/object-detection-deep-learning/MobileNetSSD_deploy.prototxt.txt', '/home/kd710306/Documents/object-detection-deep-learning/MobileNetSSD_deploy.caffemodel')
print("[INFO] computing object detections...")
vs=VideoStream(src=0).start()
time.sleep(2.0)
fps=FPS().start()
while True:	
	frame=vs.read()
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detections = net.forward()
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.2:
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
			y = startY - 15
			cv2.putText(frame, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
	cv2.imshow("Frame",frame)
	if cv2.waitKey(1)&0xFF==ord("q"):
		break
	fps.update()
fps.stop()
cv2.destroyAllWindows()
vs.stop()
