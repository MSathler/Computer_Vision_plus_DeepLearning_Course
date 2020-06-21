import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/00-puppy.jpg')
#img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)


plt.imshow(img)
plt.show()