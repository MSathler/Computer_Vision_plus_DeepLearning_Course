import cv2
import numpy as np
import matplotlib.pyplot as plt

full = cv2.imread('../Computer-Vision-with-Python/DATA/sammy.jpg')
full = cv2.cvtColor(full,cv2.COLOR_BGR2RGB)

face = cv2.imread('../Computer-Vision-with-Python/DATA/sammy_face.jpg')
face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

# All the 6 methods for comparison in a list
# Note how we are using strings, later on we'll use the eval() function to convert to function
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for m in methods:
    #CREATE A COPY
    full_coppy = full.copy()

    method = eval(m)

    #TEMPLATE MATCHING
    res = cv2.matchTemplate(full_coppy,face,method)

    min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)

    if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    height,width,channels = face.shape

    bottom_right = (top_left[0]+width, top_left[1]+height)

    cv2.rectangle(full_coppy,top_left,bottom_right,(255,0,0),10)

    #   PLOT AND SHOW IMGS
    plt.subplot(121)
    plt.imshow(res)
    plt.title('HEATMAP OF TEMPLATE MATCHING')
    plt.subplot(122)
    plt.imshow(full_coppy)
    plt.title('DETECTION OF TEMPLATE')
    plt.suptitle(m)
    plt.show()

    print('\n')
    print('\n')


