import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../DATA/rainbow.jpg',0)

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

img = cv2.imread('../DATA/crossword.jpg',0)
ret,th1 = cv2.threshold(img,180,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,8)

blended = cv2.addWeighted(src1=th1,alpha=0.6,src2=th2,beta=0.4,gamma=0)

def show_pic(img):
     fig = plt.figure(figsize=(15,15))
     ax = fig.add_subplot(111)
     ax.imshow(img,cmap='gray')

show_pic(blended)




#plt.imshow(th1,cmap='gray')
plt.show()