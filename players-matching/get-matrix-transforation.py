import cv2
import numpy as np

src=cv2.imread('src.jpg')
dst=cv2.imread('dst.jpg')

srcs=src.shape
dsts=dst.shape

pts1 = np.float32([[940,96],[1427,395],[455,395],[943,1022]])
pts2 = np.float32([[450,33],[540,300],[362,302],[450,567]])

pts3 = np.float32([[943,395]])


M = cv2.getPerspectiveTransform(pts1,pts2)
print(M)

pts3o=cv2.perspectiveTransform(pts3[None, :, :],M)


x=int(pts3o[0][0][0])
y=int(pts3o[0][0][1])
p=(x,y)


cv2.circle(dst,p, 5, (0,0,255),-1)

cv2.imshow('sada',dst)

cv2.waitKey(0)

cv2.destroyAllWindows()

