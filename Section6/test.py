
import cv2
import numpy as np
import matplotlib.pyplot as plt

mask = np.zeros((640,250))
mask =  mask[100:150,100:150]*1

plt.imshow(mask,'gray')
plt.show()

# captura = cv2.VideoCapture(0)
#
# while (captura.isOpened()):
#     ret, frame = captura.read()
#     cv2.imshow("Video", frame)
#
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# captura.release()
# cv2.destroyAllWindows()