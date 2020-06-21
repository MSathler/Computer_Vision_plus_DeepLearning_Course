import cv2
import matplotlib.pyplot as plt
import numpy as np

img1 = cv2.imread('../DATA/dog_backpack.jpg')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

#BLENDING IMAGES OF THE SAME SIZE
#img1 = cv2.resize(img1,(1200,1200))
#img2 = cv2.resize(img2,(1200,1200))

#blended = cv2.addWeighted(src1=img1,alpha=0.8,src2=img2,beta=0.2,gamma=0)

#OVERLAY SMALL IMAGE ON TOP OF A LARGER IMAGE (NO BLENDING)
#NUMPY REASSINGNMENT
img2 = cv2.resize(img2,(600,600))
large_img = img1
small_img = img2
x_offset,y_offset = 0,0
x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]
#large_img[y_offset:y_end,x_offset:x_end] = small_img

#BLEND TOGETHER IMAGES OF DIFFERENT SIZES
x_offset = img1.shape[1] - img2.shape[1]
y_offset = img1.shape[0] - img2.shape[0]
rows,cols,channels = img2.shape
roi = img1[y_offset:1401,x_offset:943]

#CREATE A MASK
img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
mask_inv = cv2.bitwise_not(img2gray)
white_background = np.full(img2.shape,255,dtype=np.uint8)
bk = cv2.bitwise_or(white_background,white_background,mask=mask_inv)
fg =cv2.bitwise_or(img2,img2,mask=mask_inv)
final_roi = cv2.bitwise_or(roi,fg)
small_img = final_roi
large_img = img1
large_img[y_offset:y_offset+small_img.shape[0],x_offset:x_offset+small_img.shape[1]] = small_img

plt.imshow(large_img)
plt.show()