import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    plt.show()

img = cv2.imread('../DATA/sudoku.jpg',0)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

laplacian = cv2.Laplacian(img,cv2.CV_64F)

blended = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

ret,th1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)

kernel = np.ones((4,4),np.uint8)
gradient = cv2.morphologyEx(blended,cv2.MORPH_GRADIENT,kernel)

display_img(gradient)