import cv2
import numpy as np

img = cv2.imread('test.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 0)
# print(img)
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

gradient_x = cv2.filter2D(img, -1, sobel_x)
gradient_y = cv2.filter2D(img, -1, sobel_y)

# print(gradient_x)
# gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
gradient_magnitude = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0)
# print(gradient_magnitude)

gradient_magnitude_display = np.uint8(gradient_magnitude)
# print(gradient_magnitude_display)
cv2.imshow('Gradient Magnitude', gradient_magnitude_display)

cv2.waitKey(0)
cv2.destroyAllWindows()
