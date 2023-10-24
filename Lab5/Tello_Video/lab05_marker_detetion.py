import cv2
import numpy as np
import time
import math
import os
from cali import Calibration
from djitellopy import Tello
from pyimagesearch.pid import PID



def main():
    # Check calibration file exists
    if not os.path.isfile("calibrate.xml"):
        print("Do calibrate first.")
        exit(1)

    f = cv2.FileStorage("calibrate.xml", cv2.FILE_STORAGE_READ)
    intrinsic = f.getNode("intrinsic").mat()
    distortion = f.getNode("distortion").mat()
    
    print(intrinsic)
    print(distortion)

    # Tello SDK
    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()
    
    # Load the predefined dictionary 
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    while True:
        frame = frame_read.frame
        
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
        modified_frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, intrinsic, distortion) 
        if np.array(rvec).ndim == 0:
            cv2.imshow("drone", modified_frame)
            key = cv2.waitKey(33)
            continue
        cv2.putText()
        modified_frame = cv2.aruco.drawAxis(modified_frame, intrinsic, distortion, rvec, tvec, 0.1)
        cv2.imshow("drone", modified_frame)
        key = cv2.waitKey(33)

    #cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

