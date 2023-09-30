import cv2 as cv
from matplotlib import pyplot as plt

# Image import - 1 for now
raw = cv.imread("Test Images/03_Control_Calcium_20x_0001.tiff", cv.IMREAD_GRAYSCALE)
assert raw is not None, "RAW file not found/read"
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
rt1, otsu = cv.threshold(raw, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Show Result
plt.imshow(otsu, "gray")
plt.show()

# Mask white cells against Original Image (LUT)
mask = cv.bitwise_and(otsu, raw)

# Show Result
plt.imshow(mask, "gray")
plt.show()

# Apply Color Map
mask2 = cv.applyColorMap(mask, cv.COLORMAP_HSV)

# Show Result
plt.imshow(mask2)
plt.show()

# Measure pixel intensities
red_hist = cv.calcHist(mask2, [0], None, [256], [0, 255])
green_hist = cv.calcHist(mask2, [1], None, [256], [0, 255])
blue_hist = cv.calcHist(mask2, [2], None, [256], [0, 255])

# Plot graph
plt.subplot(4, 1, 1)
plt.imshow(mask2)
plt.title('image')
plt.xticks([])
plt.yticks([])

plt.subplot(4, 1, 2)
plt.plot(red_hist, color='r')
plt.xlim([0, 255])
plt.title('red histogram')

plt.subplot(4, 1, 3)
plt.plot(green_hist, color='g')
plt.xlim([0, 255])
plt.title('green histogram')

plt.subplot(4, 1, 4)
plt.plot(blue_hist, color='b')
plt.xlim([0, 255])
plt.title('blue histogram')

plt.tight_layout()
plt.show()
