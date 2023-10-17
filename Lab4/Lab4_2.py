import cv2
import numpy as np

h, w = 640, 480

background = cv2.imread("screen.jpg")

cap_corner = np.float32([(0, 0), (h-1, 0), (h-1, w-1), (0, w-1)])
img_corner = np.float32([(275, 188), (618, 85), (624, 376), (262, 406)])

proj = cv2.getPerspectiveTransform(cap_corner, img_corner)
cap = cv2.VideoCapture(0)
res = np.ones((3, h*w))
for i in range(h):
    for j in range(w):
        res[0][i*w+j] = i
        res[1][i*w+j] = j
        res[2][i*w+j] = 1
res = proj @ res
while True:
    _, frame = cap.read()
    for i in range(h):
        for j in range(w):
            x = int(res[0, w*i+j] / res[2, w*i+j])
            y = int(res[1, w*i+j] / res[2, w*i+j])
            background[y][x][:] = frame[j][i][:]

    cv2.imshow("image", background)
    cv2.waitKey(500)
# print(frame.shape)