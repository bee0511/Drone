import cv2
import numpy as np

input_image = cv2.imread('histogram.jpg')

b, g, r = cv2.split(input_image)

equalized_b = cv2.equalizeHist(b)
equalized_g = cv2.equalizeHist(g)
equalized_r = cv2.equalizeHist(r)

equalized_image = cv2.merge([equalized_b, equalized_g, equalized_r])

cv2.imshow('Original Image', input_image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
