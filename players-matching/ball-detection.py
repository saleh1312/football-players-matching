import cv2
import numpy as np


cap = cv2.VideoCapture(r'C:\Users\iP\Desktop\proj\data\video.avi')
temp=cv2.imread('temp5.jpg',0)
w, h = temp.shape[::-1]


if (cap.isOpened()== False): 
    print("Error opening video stream or file")
    
i=0
while(cap.isOpened()):
    
  
    ret, img = cap.read()
    if(i <310):
        i=i+1
        continue
    if ret == True :
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,temp,cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img,top_left, bottom_right, (0,255,100), 2)
        
        cv2.imshow('sads',img)
        i=i+1
            
    if cv2.waitKey(0)== 27:
        
        break
    
print(i)
cap.release()
cv2.destroyAllWindows()
