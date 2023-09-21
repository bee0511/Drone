import cv2
import numpy as np
import os

# Define a constant to specify how much the image should be enlarged
ENLARGE_SIZE = 3

# Define the filename of the image to be processed
filename = './test.jpg'

# Read the image from the specified filename
img = cv2.imread(filename)

# Create a new image with larger dimensions, initialized as black (zeros)
new_img = np.zeros((img.shape[0] * ENLARGE_SIZE, img.shape[1] * ENLARGE_SIZE, 3), dtype=np.uint8)

# Calculate scaling factors for mapping pixels from the original image to the new image
# -1 is to avoild boundary conditions
m = new_img.shape[0] / (img.shape[0] - 1)
n = new_img.shape[1] / (img.shape[1] - 1)

# Loop through each pixel in the new image
for i in range(new_img.shape[0]):
    for j in range(new_img.shape[1]):
        # Calculate the corresponding pixel coordinates in the original image
        y = int(i / m)
        x = int(j / n)
        
        # Calculate the fractional parts of the coordinates
        a = (i - y * m) / m
        b = (j - x * n) / n
        
        # Perform bilinear interpolation to determine the color of the new pixel
        new_img[i, j] = ((1 - a) * (1 - b) * img[y, x]) + a * (1 - b) * img[y + 1, x] + b * (1 - a) * img[y, x + 1] + a * b * img[y + 1, x + 1]

# Create a named window for displaying the new image
winname = "Bilinear"
cv2.namedWindow(winname)

# Move the window to a specific position on the screen
cv2.moveWindow(winname, 40, 30)

# Display the new image in the window
cv2.imshow(winname, new_img)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the enlarged image to a file
cv2.imwrite('Bilinear.jpg', new_img)