import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from numpy import argmax


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
model = load_model(r'model.h5')



classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("3.jpg")

copy=img.copy()
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[class_ids[i]]
        if label=='person':
            roi = img[y:y+h,x:x+w]
            roi=cv2.resize(roi, (96,96))
            ym=model.predict(np.reshape(roi,(1,96,96,3)))
            ym=argmax(ym)
            
            if ym==0:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,0,255), 2)
            elif ym==1:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,255,0), 2)
            elif ym==2:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (255,0,0), 2)
           
            
            



cv2.imshow("Image", copy)
cv2.waitKey(0)
cv2.destroyAllWindows()