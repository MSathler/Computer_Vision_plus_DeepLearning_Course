import cv2
import numpy as np
import matplotlib.pyplot as plt

dark_horse = cv2.imread('../DATA/horse.jpg') #ORIGINAL BGR OPENCV
show_horse = cv2.cvtColor(dark_horse,cv2.COLOR_BGR2RGB)                  #CONVERTED TO RGB FOR SHOW

rainbow = cv2.imread('../DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow,cv2.COLOR_BGR2RGB)

blue_bricks = cv2.imread('../DATA/bricks.jpg')
show_bricks = cv2.cvtColor(blue_bricks,cv2.COLOR_BGR2RGB)

#OPENCV BGR
hist_values = cv2.calcHist([blue_bricks],channels=[0],mask=None,histSize=[256],ranges=[0,256])
hist_values = cv2.calcHist([dark_horse],channels=[0],mask=None,histSize=[256],ranges=[0,256])

img = dark_horse
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])


#plt.plot(hist_values)
plt.imshow(mask,cmap='gray')
plt.show()