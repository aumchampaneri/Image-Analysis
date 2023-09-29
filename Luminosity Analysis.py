import cv2 as cv
from matplotlib import pyplot as plt

# Image import - 1 for now
img = cv.imread("Test Images/03_Control_Calcium_20x_0001.tiff", cv.IMREAD_GRAYSCALE)
assert img is not None, "File not found/read"
# TODO add remaining files

"""
Pipeline:
1. Otsu Thresholding
2. Mask Whiie Cells against original image
3. Pixel Intensity Plot per image
    3a. Average image intensity per group
4. Apply to the 40x images
"""

# Otsu Thresholding
ret1, th1 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Show Result
plt.imshow(th1, "gray")
plt.show()
