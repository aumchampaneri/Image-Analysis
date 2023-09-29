import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("03_Control_Calcium_20x_0000.tiff", cv.IMREAD_GRAYSCALE)
assert img is not None, "File not found/read"

#
