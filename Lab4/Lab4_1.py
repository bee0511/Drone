import cv2
import time
import numpy as np


class Calibration:
    NX, NY = 9, 6
    WIN_SIZE = (11, 11)
    ZERO_ZONE = (-1, -1)
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    def __init__(self, drone):
        self.object_points = []
        self.img_points = []
        self.camera_matrix, self.distortion_coeffs = None
        self.drone = drone

    def do_cali(self):
        object_point = np.zeros((9 * 6, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        while(len(self.object_points) < 50):
            frame_read = self.drone.get_frame_read()
            frame = frame_read.frame
            grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            retCorner, corners = cv2.findChessboardCorners(grayed_frame, (self.NX, self.NY), None)

            if retCorner:
                corner2 = cv2.cornerSubPix(grayed_frame, corners, self.WIN_SIZE, self.ZERO_ZONE, self.CRITERIA)
                self.object_points.append(object_point.copy())
                self.img_points.append(corner2)
                cv2.imshow('grayed_frame', grayed_frame)
                cv2.drawChessboardCorners(grayed_frame, (self.NX, self.NY), corners, ret)
                cv2.waitKey(100)
            time.sleep(0.1)
        ret, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points,
            self.img_points, 
            (self.NX, self.NY), 
            None, 
            None
        )

        self.distortion_coeffs = distortion_coeffs
        self.camera_matrix = camera_matrix

        return (camera_matrix, distortion_coeffs)