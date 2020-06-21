import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('../Computer-Vision-with-Python/DATA/flat_chessboard.png')

found,corners = cv2.findChessboardCorners(flat_chess,(7,7))

cv2.drawChessboardCorners(flat_chess,(7,7),corners,found)

dots = cv2.imread('../Computer-Vision-with-Python/DATA/dot_grid.png')

found2,corners2 = cv2.findCirclesGrid(dots,(10,10),cv2.CALIB_CB_SYMMETRIC_GRID)

cv2.drawChessboardCorners(dots,(10,10),corners2,found2)


plt.imshow(dots)
plt.show()
