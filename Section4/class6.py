import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_img():
    black_img = np.zeros((600,600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(black_img,text='ABCDE',org=(50,400),fontFace=font,fontScale=5,color=(255,255,255),thickness=20)
    return black_img

def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    plt.show()

img = load_img()
kernel = np.ones((5,5),dtype=np.uint8)
result = cv2.erode(img,kernel,iterations=2)
white_noise = np.random.randint(low=0,high=2,size=(600,600))*255
noise_img = white_noise + img
opening = cv2.morphologyEx(noise_img,cv2.MORPH_OPEN,kernel)
black_noise = np.random.randint(low=0,high=2,size=(600,600))
black_noise = black_noise * -255
black_noise_img = img + black_noise
black_noise_img[black_noise_img == -255] = 0
closing = cv2.morphologyEx(black_noise_img,cv2.MORPH_CLOSE,kernel)
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)



display_img(gradient)