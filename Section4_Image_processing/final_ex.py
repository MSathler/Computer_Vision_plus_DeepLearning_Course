import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('../DATA/giraffes.jpg')
#fix_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
#ret,thres = cv2.threshold(fix_img,thresh=127,maxval=255,type=cv2.THRESH_BINARY)

kernel = np.ones([4,4],dtype=np.float32) / 10
#result = cv2.filter2D(fix_img,-1,kernel)

#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
color = ['b','g','r']
for i,col in enumerate(color):
    hist = cv2.calcHist([img],[i],mask=None,histSize=[256],ranges=[0,256])
    plt.plot(hist)
#plt.imshow(sobelx,cmap='gray')
plt.show()