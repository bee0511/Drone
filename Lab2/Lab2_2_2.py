import cv2
import numpy as np

input_image = cv2.imread('histogram.jpg')

hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv_image)

equalized_v = cv2.equalizeHist(v)

equalized_hsv_image = cv2.merge([h, s, equalized_v])

equalized_image = cv2.cvtColor(equalized_hsv_image, cv2.COLOR_HSV2BGR)

cv2.imshow('Original Image', input_image)
cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()