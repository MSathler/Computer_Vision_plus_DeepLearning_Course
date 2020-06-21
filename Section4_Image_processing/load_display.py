import cv2
import matplotlib.pyplot as plt
import numpy as np

class img():
    def load_img():
        img = cv2.imread('../DATA/bricks.jpg').astype(np.float32)/255
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img

    def display_img(img):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        plt.show()
