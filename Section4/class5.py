import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_img():
    img = cv2.imread('../Computer-Vision-with-Python/DATA/bricks.jpg').astype(np.float32)/255
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()
gamma = 1/4
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,text='bricks',org=(10,600),fontFace=font,fontScale=10,color=(255,0,0),thickness=4)
#result = np.power(i,gamma)
blurred = cv2.blur(img,ksize=(5,10))
blurred_img = cv2.GaussianBlur(img,(5,5),10)
median_result = cv2.medianBlur(img,5)
blur = cv2.bilateralFilter(img,9,75,75)
kernel = np.ones(shape=(5,5),dtype=np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

##PART 2 NEW IMG

img = cv2.imread('../Computer-Vision-with-Python/DATA/sammy.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
noise_img = cv2.imread('../Computer-Vision-with-Python/DATA/sammy_noise.jpg')
media = cv2.medianBlur(noise_img,5)

display_img()