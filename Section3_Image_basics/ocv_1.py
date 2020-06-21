import numpy as np
import matplotlib.pyplot as plt

import cv2

img = cv2.imread('../DATA/00-puppy.jpg')
if img is None:
    print("Caminho errado, tente novamente")

print("Imagem valida")
img.shape

#MATPLOTLIB --> RGB
#OPENCV     --> BGR
fix_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img_gray = cv2.imread('../DATA/00-puppy.jpg', cv2.IMREAD_GRAYSCALE)
new_img = cv2.resize(fix_img,(1000,400))
w_ratio = 0.5
h_ratio = 0.5
new_img2 = cv2.resize(fix_img,(0,0),fix_img,w_ratio,h_ratio)
new_img3 = cv2.flip(fix_img,1)
cv2.imwrite('../totally_new.jpg', fix_img)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ax.imshow(fix_img)

plt.imshow(new_img3)
plt.show()