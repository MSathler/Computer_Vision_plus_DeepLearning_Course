import numpy as np
import matplotlib.pyplot as plt
import cv2
img = cv2.imread('../Computer-Vision-with-Python/DATA/00-puppy.jpg')
while True:
    cv2.imshow('Puppy',img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()