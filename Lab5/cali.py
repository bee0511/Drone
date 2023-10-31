import cv2
import time
import numpy as np
from djitellopy import Tello


class Calibration:
    NX, NY = 9, 6
    WIN_SIZE = (11, 11)
    ZERO_ZONE = (-1, -1)
    CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    def __init__(self):
        self.object_points = []
        self.img_points = []
        self.camera_matrix, self.distortion_coeffs = None, None

    def do_cali(self, frame_read):
        object_point = np.zeros((9 * 6, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        key = -1

        while len(self.object_points) < 12:
            frame = frame_read.frame
            cv2.imshow('frame', frame)
            grayed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            retCorner, corners = cv2.findChessboardCorners(grayed_frame, (self.NX, self.NY), None)
            if retCorner and key != -1:
                corner2 = cv2.cornerSubPix(grayed_frame, corners, self.WIN_SIZE, self.ZERO_ZONE, self.CRITERIA)
                self.object_points.append(object_point.copy())
                self.img_points.append(corner2)
                cv2.drawChessboardCorners(grayed_frame, (self.NX, self.NY), corners, True)
                cv2.imshow('grayed_frame', grayed_frame)
                print('[*] Captured Object', len(self.object_points))
                # cv2.imwrite(f'./saved_{str(self.object_points)}.png', grayed_frame)
            key = cv2.waitKey(50)

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
    
if __name__ == '__main__':
    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()
    cali = Calibration()
    cameraMatrix, distCoeffs = cali.do_cali(frame_read)

    f = cv2.FileStorage("calibrate.xml", cv2.FILE_STORAGE_WRITE)
    f.write("intrinsic", cameraMatrix)
    f.write("distortion", distCoeffs)
    f.release()
    
    drone.streamoff()