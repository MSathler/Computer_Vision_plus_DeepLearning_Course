import cv2
import matplotlib.pyplot as plt
import numpy as np



img = cv2.imread('../Computer-Vision-with-Python/DATA/dog_backpack.jpg')

img_arr = np.asarray(img)
img_arr_fix = cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)
rotate_img_arr_r = cv2.flip(img_arr_fix,1)
cv2.rectangle(img_arr_fix,pt1=(250,400),pt2=(600,700),color=(255,0,0),thickness=5)
vertices = np.array([ [250,700], [650,700], [480,400]],np.int32)
pts = vertices.reshape((-1,1,2))
cv2.polylines(img_arr_fix,[pts],isClosed=True,color=(0,0,255),thickness=5)
#cv2.fillPoly(img_arr_fix,[pts],(0,0,255))


plt.imshow(img_arr_fix)
plt.show()