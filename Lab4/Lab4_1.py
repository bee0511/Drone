import cv2
import time
import numpy as np
cap = cv2.VideoCapture(0) #device
nx= 9 
ny= 6 
winSize = (11, 11)
zeroZone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

objp = np.zeros((9*6,3), np.float32)
objp[:, :2]=np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objectPoint = []
imgPoint = []

while(len(objectPoint) < 50):
    ret, frame = cap.read()
    #ret is True if read() successed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    retCorner, corners = cv2.findChessboardCorners(frame, (nx, ny), None )
    print ( 'ret:' ,retCorner)
    # print(len(corners))

    # If found, draw corners 
    if retCorner:
     # Draw and display the corners
        # cv2.drawChessboardCorners(frame, (nx, ny), corners, ret)
        corner2 = cv2.cornerSubPix(frame, corners, winSize, zeroZone, criteria)
        # print(corner2)
        objectPoint.append(objp.copy())
        imgPoint.append(corner2)
        cv2.imshow('frame', frame)
        cv2.drawChessboardCorners(frame, (nx, ny), corners, ret)
        cv2.waitKey(100)
    time.sleep(0.1)
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objectPoint, imgPoint, (nx, ny), None, None)
f = cv2.FileStorage("calibrate.xml", cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", cameraMatrix)
f.write("distortion", distCoeffs)
f.release()