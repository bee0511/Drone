import numpy as np
import cv2

def otsu(img, th):
    # create the thresholded image
    thresholded_im = np.zeros(img.shape)
    thresholded_im[img >= th] = 1

    # compute weights
    pixels = img.size
    nonzero_pixels = np.count_nonzero(thresholded_im)
    weight1 = nonzero_pixels / pixels
    weight0 = 1 - weight1

    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = img[thresholded_im == 1]
    val_pixels0 = img[thresholded_im == 0]

    # compute variance of these classes
    var0 = var1 = 0
    if len(val_pixels0): 
        var0 = np.var(val_pixels0)
    if len(val_pixels1): 
        var1 = np.var(val_pixels1)

    return weight0 * var0 + weight1 * var1

img = cv2.imread("otsu.jpg")

# compute otsu criteria
criterias = [0]*255
for i in range(255):
    criterias[i] = otsu(img, i)

# best threshold = min otsu criteria
best_threshold = np.argmin(criterias)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if np.mean(img[i][j]) > best_threshold:
            img[i][j] = np.array([255, 255, 255])
        else:
            img[i][j] = np.zeros(3)

cv2.imshow("My Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()