import cv2
import numpy as np
import matplotlib.pyplot as plt

gorilla = cv2.imread('../Computer-Vision-with-Python/DATA/gorilla.jpg',0)
color_gorilla = cv2.imread('../Computer-Vision-with-Python/DATA/gorilla.jpg')
show_gorilla = cv2.cvtColor(color_gorilla,cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(color_gorilla,cv2.COLOR_BGR2HSV)
hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
eq_color_gorilla = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

eq_gorilla = cv2.equalizeHist(gorilla)
hist_values = cv2.calcHist([eq_gorilla],channels=[0],mask=None,histSize=[256],ranges=[0,256])


plt.plot(hist_values)
plt.show()
