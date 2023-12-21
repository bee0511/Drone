import numpy as np
import cv2
import os
import time
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
from otsu import *

KERNEL_SIZE = 3
CALIBRATE_FILE = "calibrate-01.xml"
MAX_SPEED_THRESHOLD = 25
HORIZONTAL_OFFSET = 25
VERTICAL_UPWARD_OFFSET = 15
VERTICAL_DOWNWARD_BOOST = 60

SCALING_FACTOR_YAW = 1.2

# Check calibration file exists
if not os.path.isfile(CALIBRATE_FILE):
    print("Do calibrate first.")
    exit(1)

f = cv2.FileStorage(CALIBRATE_FILE, cv2.FILE_STORAGE_READ)
intrinsic = f.getNode("intrinsic").mat()
distortion = f.getNode("distortion").mat()

# Load the predefined dictionary
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Initialize the detector parameters using default values
parameters = cv2.aruco.DetectorParameters_create()

pid_horizontal = PID(kP=0.7, kI=0.0001, kD=0.1)
pid_vertical = PID(kP=0.7, kI=0.0001, kD=0.1)
pid_z = PID(kP=0.7, kI=0.0001, kD=0.1)
pid_yaw = PID(kP=0.7, kI=0.0001, kD=0.1)
height_controller_pid = PID(kP=0.7, kI=0.0001, kD=0.1)

def clamping(val):
    if val > MAX_SPEED_THRESHOLD:
        return MAX_SPEED_THRESHOLD
    elif val < -MAX_SPEED_THRESHOLD:
        return -MAX_SPEED_THRESHOLD
    return int(val)

def tracking_aruco(frame, marker_id, z_distance):
    # Make a copy of frame
    frame = frame.copy()
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        frame, dictionary, parameters=parameters
    )
    modified_frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
        markerCorners, 15, intrinsic, distortion
    )
    z_update = None
    v_update = None
    yaw_update = None

    # If we see some markers
    if np.array(rvec).ndim == 0:
        return None

    # Iterate all markers
    for i in range(rvec.shape[0]):
        modified_frame = cv2.aruco.drawAxis(
            modified_frame,
            intrinsic,
            distortion,
            rvec[i, :, :],
            tvec[i, :, :],
            10,
        )
        cv2.putText(
            frame,
            f"x = {tvec[0,0,0]}, y = {tvec[0,0,1]}, z = {tvec[0,0,2]}",
            (100, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        id = markerIds[i][0]
        if id != marker_id:
            continue

        rotation_matrix = np.zeros(shape=(3, 3))
        cv2.Rodrigues(rvec[i], rotation_matrix, jacobian=0)
        ypr = cv2.RQDecomp3x3(rotation_matrix)[0]
        z_error = tvec[0, 0, 2] - z_distance
        yaw_error = ypr[1] * SCALING_FACTOR_YAW
        horizontal_error = tvec[0, 0, 0] - HORIZONTAL_OFFSET
        vertical_error = tvec[0, 0, 1]
        if vertical_error - VERTICAL_UPWARD_OFFSET < 0:
            y_error = -(vertical_error - VERTICAL_UPWARD_OFFSET)
        else:
            y_error = vertical_error - VERTICAL_DOWNWARD_BOOST

        h_update = clamping(pid_horizontal.update(horizontal_error, sleep=0))
        v_update = clamping(pid_vertical.update(y_error, sleep=0))
        z_update = clamping(pid_z.update(z_error, sleep=0))
        yaw_update = clamping(pid_yaw.update(yaw_error, sleep=0))
        return (modified_frame, h_update, v_update, z_update, yaw_update, tvec[0, 0, 2])

    return None


def send_drone_control(drone, key, x_update, y_update, z_update, yaw_update):
    if key != -1:
        keyboard(drone, key)
    else:
        drone.send_rc_control(x_update, z_update, y_update, yaw_update)
        print(x_update, yaw_update, z_update, yaw_update)

class LineFollower:
    def __init__(self, drone, path):
        self.threshold = 150
        self.drone = drone
        self.otsu_threshold = OtsuThreshold()
        self.speed = 10
        self.full_path = path
        self.current_path = 0

    def scan_result(self, frame):
        '''
        Seperate frame into 4 parts by current direction, if right/left, then seperate by height, if up/down, then seperate by width
        '''
        # 720 x 960, 180 x 240
        frame_height, frame_width, _ = frame.shape

        # Seperate frame into 4 parts, each part is a sensor, then average each sensor
        # 0-20%: sensor_1
        # 25-45%: sensor_2
        # 50-70%: sensor_3
        # 80%-100%: sensor_4
        if self.full_path[self.current_path] == 'right':
            sensor_size = 70

            sensor_1 = frame[
                0: sensor_size, 
                (frame_width // 4) * 3:frame_width
            ]
            sensor_2 = frame[
                sensor_size + 60: sensor_size * 2 + 60,
                (frame_width // 4) * 3:frame_width
            ]
            sensor_3 = frame[
                frame_height - sensor_size * 2 - 60: frame_height - sensor_size - 60, 
                (frame_width // 4) * 3:frame_width
            ]
            sensor_4 = frame[
                frame_height - sensor_size: frame_height,
                (frame_width // 4) * 3:frame_width
            ]
        elif self.full_path[self.current_path] == 'left':
            sensor_size = 70

            sensor_1 = frame[
                0: sensor_size, 
                :frame_width // 4
            ]
            sensor_2 = frame[
                sensor_size + 60: sensor_size * 2 + 60,
                :frame_width // 4
            ]
            sensor_3 = frame[
                frame_height - sensor_size * 2 - 60: frame_height - sensor_size - 60, 
                :frame_width // 4
            ]
            sensor_4 = frame[
                frame_height - sensor_size: frame_height,
                :frame_width // 4
            ]
        elif self.full_path[self.current_path] == 'up':
            sensor_size = 100
            
            sensor_1 = frame[
                : frame_height // 4,
                0: sensor_size
            ]
            sensor_2 = frame[
                : frame_height // 4,
                sensor_size + 120: sensor_size * 2 + 120,
            ]
            sensor_3 = frame[
                : frame_height // 4,
                frame_width - sensor_size * 2 - 120: frame_width - sensor_size - 120, 
            ]
            sensor_4 = frame[
                : frame_height // 4,
                frame_width - sensor_size: frame_width,
            ]
        elif self.full_path[self.current_path] == 'down':
            sensor_size = 100
            
            sensor_1 = frame[
                (frame_height // 4) * 3: frame_height,
                0: sensor_size
            ]
            sensor_2 = frame[
                (frame_height // 4) * 3: frame_height,
                sensor_size + 120: sensor_size * 2 + 120,
            ]
            sensor_3 = frame[
                (frame_height // 4) * 3: frame_height,
                frame_width - sensor_size * 2 - 120: frame_width - sensor_size - 120, 
            ]
            sensor_4 = frame[
                (frame_height // 4) * 3: frame_height,
                frame_width - sensor_size: frame_width,
            ]
        else:
            return 0, 0, 0, 0
        
        average_1 = np.average(sensor_1)
        average_2 = np.average(sensor_2)
        average_3 = np.average(sensor_3)
        average_4 = np.average(sensor_4)
        cv2.imshow('sensor_1', sensor_1)
        cv2.imshow('sensor_2', sensor_2)
        cv2.imshow('sensor_3', sensor_3)
        cv2.imshow('sensor_4', sensor_4)
        return average_1, average_2, average_3, average_4


    def follow_line(self, frame):
        processed_frame = self.otsu_threshold.process_frame(frame)

        # Scan result
        average_1, average_2, average_3, average_4 = self.scan_result(processed_frame)
        
        triggered_1 = True if average_1 <= self.threshold else False
        triggered_2 = True if average_2 <= self.threshold else False
        triggered_3 = True if average_3 <= self.threshold else False
        triggered_4 = True if average_4 <= self.threshold else False
        print(average_1, average_2, average_3, average_4)
        print(self.full_path[self.current_path], triggered_1, triggered_2, triggered_3, triggered_4)
        # Left/Right Sensor Layout, Top/Down
        #   |1|                     |1|2|3|4|
        #   |2|
        #   |3|
        #   |4|

        cv2.imshow("line_follower__frame", processed_frame)


def main():
    pid_horizontal.initialize()
    pid_vertical.initialize()
    pid_z.initialize()
    pid_yaw.initialize()
    # Tello SDK
    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()
    key = -1
    distance_to_target = 1e9
    drone.send_rc_control(0, 0, 0, 0)

    # Aruco Tracking to specific range
    TARGET_DISTANCE = 40
    while distance_to_target > TARGET_DISTANCE:
        frame = frame_read.frame
        key = cv2.waitKey(5)
        cv2.imshow("frame", frame)

        tracking_result = tracking_aruco(frame, 3, TARGET_DISTANCE)
        if key != -1:
            keyboard(drone, key)
        elif tracking_result:
            (
                labeled_frame,
                x_update,
                y_update,
                z_update,
                yaw_update,
                distance_to_target,
            ) = tracking_result
            print(distance_to_target)
            send_drone_control(drone, key, x_update, y_update, z_update, 0) 
            cv2.putText(
                labeled_frame,
                f"{x_update}, {y_update}, {z_update}, {yaw_update}",
                (200, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("aruco", labeled_frame)
    
    drone.send_rc_control(0, 0, 0, 0)
    # Line Following
    path = (
        'right',
        'up',
        'right',
        'up',
        'left',
        'down'
    )
    line_follower = LineFollower(drone, path)
    while True:
        frame = drone.get_frame_read().frame
        key = cv2.waitKey(5)
        if key != -1:
            keyboard(drone, key)
            continue
        else:
            line_follower.follow_line(frame)

if __name__ == "__main__":
    main()
