import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('../Computer-Vision-with-Python/DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)

gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)
real_chess = cv2.imread('../Computer-Vision-with-Python/DATA/real_chessboard.jpg')

real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)
gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray_real_chess)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
real_chess[dst>0.01*dst.max()] = [255,0,0]

plt.imshow(real_chess)
plt.show()