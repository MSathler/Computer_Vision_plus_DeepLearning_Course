import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

pic = Image.open('../DATA/00-puppy.jpg')

pic_arr = np.asarray(pic)
pic_red = pic_arr.copy()

#RED CHANNEL BALUES 0-255
#plt.imshow(pic_red[:,:,0],cmap='gray')

pic_red[:,:,0] = 0
pic_red[:,:,1] = 0
plt.imshow(pic_red)
plt.show()