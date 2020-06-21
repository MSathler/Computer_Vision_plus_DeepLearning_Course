import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/sammy_face.jpg')
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

med_val = np.median(img)
#LOWER THRESHOLD TO EITHER 0 OR 70% OF THE MEDIAN BALUE WHICHEVER IS GREATER
lower = int(max(0,0.7*med_val))
# UPPER THRESHOLD TO EITHER 130% OF THE MEDIAN OR THE MAX 255, EHICHEVER IS SMALLER
upper =  int(min(255,1.3*med_val))
blurred_img = cv2.blur(img,(5,5))
edges = cv2.Canny(blurred_img,lower,upper+50)
plt.imshow(edges,'gray')
plt.show()