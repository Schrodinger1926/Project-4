import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

offset = 100

"""
src = np.float([[370, 610],
                [959, 626],
                [557, 477],
                [727, 477]])

src = np.float32([[557, 477],
                  [370, 610],
                  [959, 626],
                  [727, 477]])

dst = np.float32([[offset, offset],
                  [img_size[0]-offset, offset],
                  [img_size[0]-offset, img_size[1]-offset],
                  [offset, img_size[1]-offset]])
"""

def get_wraped_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])
    img_size = (420 , 420)

    offset = 100 # offset for dst points

    src1 = np.float32([[557, 477],
                       [370, 610],
                       [959, 626],
                       [727, 477]])

    # deep top
    src2 = np.float32([[605, 444],
                       [370, 610],
                       [959, 626],
                       [676, 446]])

    # new data
    src3 = np.float32([[613, 442],
                       [411, 580],
                       [895, 585],
                       [663, 440]])

    dst = np.float32([[offset, offset],
                      [offset, img_size[1]-offset],
                      [img_size[0]-offset, img_size[1]-offset],
                      [img_size[0]-offset, offset]])

    M = cv2.getPerspectiveTransform(src1, dst)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, img_size)

    return warped[:318, :, :]


DIR = 'test_images'

for filename in os.listdir(DIR):
    image = cv2.imread(os.path.join(DIR, filename))
    warped_image = get_wraped_image(image)
    cv2.imwrite(os.path.join('rnd', 'warped', 'color_{}'.format(filename)), warped_image)
