import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from numpy import argmax


model = load_model(r'files\model.h5')

cap = cv2.VideoCapture(r'data\video.avi')
temp=cv2.imread(r'data\temp.jpg',0)
ground=cv2.imread(r'data\dst.jpg')

wt, ht = temp.shape[::-1]

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#if you want to write video
#out = cv2.VideoWriter('match.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1920,1080))
#out2 = cv2.VideoWriter('plane.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (900,600))

if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    
# Load Yolo
net = cv2.dnn.readNet("files\yolov3.weights", "files\yolov3.cfg")
classes = []
with open("files\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3)) 
    

def plane(players,ball):
    coptemp=ground.copy()
    matrix=np.array([[ 2.56945407e-01,  5.90910632e-01,  1.94094537e+02],
                     [-1.33508274e-02,  1.37658562e+00, -8.34967286e+01],
                     [-3.41878940e-05,  1.31509536e-03,  1.00000000e+00]])
    
    for p in players:
        x=p[0]+int(p[2]/2)
        y=p[1]+p[3]
        pts3 = np.float32([[x,y]])
        pts3o=cv2.perspectiveTransform(pts3[None, :, :],matrix)
        x1=int(pts3o[0][0][0])
        y1=int(pts3o[0][0][1])
        pp=(x1,y1)
        if(p[4]==0):
            cv2.circle(coptemp,pp, 15, (255,0,0),-1)
        elif p[4]==1:
            cv2.circle(coptemp,pp, 15, (255,255,255),-1)
        elif p[4]==2:
            #print hakm
            #cv2.circle(coptemp,pp, 15, (0,0,255),-1)
            pass
    if len(ball) !=0:
        
        xb=ball[0]+int(ball[2]/2)
        yb=ball[1]+int(ball[3]/2)
        pts3ball = np.float32([[xb,yb]])
        pts3b=cv2.perspectiveTransform(pts3ball[None, :, :],matrix)
        x2=int(pts3b[0][0][0])
        y2=int(pts3b[0][0][1])
        pb=(x2,y2)
        cv2.circle(coptemp,pb, 15, (0,0,0),-1)
    return coptemp


def get_players(outs,height, width):
    class_ids = []
    confidences = []
    boxes = []
    players=[]
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
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label=='person':
                players.append(boxes[i])
            
    return players



i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    players=[]
    ball=[]
    if i<310:
        i=i+1
        continue
    
    if ret == True :
        copy=frame.copy()
        gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        height, width, channels = frame.shape
        
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)
        outs=get_players(outs, height, width)
        for i in range(len(outs)):
            x, y, w, h = outs[i]
            roi = frame[y:y+h,x:x+w]
            
            #some frames are bad so resize function throw an error
            try:
                roi=cv2.resize(roi, (96,96))
            except:
                continue
            ym=model.predict(np.reshape(roi,(1,96,96,3)))
            ym=argmax(ym)
            
            players.append([x,y,w,h,ym])
            
            if ym==0:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,0,255), 2)
            elif ym==1:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (0,255,0), 2)
            elif ym==2:
                cv2.rectangle(copy, (x, y), (x + w, y + h), (255,0,0), 2)
            
        
        res = cv2.matchTemplate(gray,temp,cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if min_val < 0.05:
            top_left = min_loc
            bottom_right = (top_left[0] + wt, top_left[1] + ht)
            ball.append(top_left[0])
            ball.append(top_left[1])
            ball.append(wt)
            ball.append(ht)
            cv2.rectangle(copy,top_left, bottom_right, (0,255,100), 2)
            
        p=plane(players, ball)
            
        #out.write(frame)
        #out2.write(p)
        cv2.imshow('img',copy)
        cv2.imshow('plane',p)
        
    # this will run the video without stop and maybe the cv2 window will stop between every frame
    # depending on your pc power ( i recommend to use (opencv with gpu) and colab to run script quickly ) 
    # if you want script stop between every frame and manually you allow the script to continue change it t ocv2.waitKey(0)
    if cv2.waitKey(1)==27:
        
        break
    

# When everything done, release the video capture object
cap.release()
#out.release()
#out2.release()
# Closes all the frames
cv2.destroyAllWindows()

















