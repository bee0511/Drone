import numpy as np
import cv2
import os
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard
from otsu import *

KERNEL_SIZE = 3
CALIBRATE_FILE = "calibrate-01.xml"
THRESHOLD = 150
MAX_SPEED_THRESHOLD = 30

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

pid_controller_x = PID(kP=0.4, kI=0.0008, kD=0.4)
pid_controller_y = PID(kP=0.6, kI=0.0002, kD=0.1)
pid_controller_z = PID(kP=0.5, kI=0.0002, kD=0.1)
yaw_controller_pid = PID(kP=0.7, kI=0.0002, kD=0.1)
height_controller_pid = PID(kP=0.7, kI=0.0001, kD=0.1)


def clamping(val):
    if val > MAX_SPEED_THRESHOLD:
        return MAX_SPEED_THRESHOLD
    elif val < -MAX_SPEED_THRESHOLD:
        return -MAX_SPEED_THRESHOLD
    return int(val)


K_X_OFFSET = 30
K_Y_UPWARD_OFFSET = 15
K_Y_DOWNWARD_BOOST = 60


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
    y_update = None
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
        yaw_error = ypr[1]
        x_error = tvec[0, 0, 0] - K_X_OFFSET
        if tvec[0, 0, 1] - K_Y_UPWARD_OFFSET < 0:
            y_error = -(tvec[0, 0, 1] - K_Y_UPWARD_OFFSET)
        else:
            y_error = tvec[0, 0, 1] - K_Y_DOWNWARD_BOOST
        z_error = tvec[0, 0, 2] - z_distance

        x_update = clamping(pid_controller_x.update(x_error, sleep=0))
        y_update = clamping(pid_controller_y.update(y_error, sleep=0))
        z_update = clamping(pid_controller_z.update(z_error, sleep=0))
        yaw_update = clamping(yaw_controller_pid.update(-(yaw_error + 3), sleep=0))
        return (modified_frame, x_update, y_update, z_update, yaw_update, tvec[0, 0, 2])
    return None


def send_drone_control(drone, key, x_update, y_update, z_update, yaw_update):
    if key != -1:
        keyboard(drone, key)
    elif x_update and y_update and z_update and yaw_update:
        # drone.send_rc_control(x_update, z_update, y_update, 0)
        drone.send_rc_control(x_update, z_update, y_update, yaw_update)
        print(x_update, yaw_update, z_update, yaw_update)


class LineFollower:
    PATH = (
        'right',
        'up',
        'right',
        'up',
        'left',
        'down'
    )

    def __init__(self, drone):
        self.drone = drone
        self.otsu_threshold = OtsuThreshold()
        self.speed = 20
        self.current_direction = 'left'

    def scan_result(self, frame):
        '''
        Seperate frame into 4 parts by current direction, if right/left, then seperate by height, if up/down, then seperate by width
        '''
        # 720 x 960
        frame_height, frame_width, _ = frame.shape
        if self.current_direction == 'right':
            unit = frame_width // 4
            sensor_1 = frame[:unit - 60, frame_width // 3: frame_width // 3 * 2]
            sensor_2 = frame[unit: unit * 2, frame_width // 3: frame_width // 3 * 2]
            sensor_3 = frame[unit * 2: unit * 3, frame_width // 3: frame_width // 3 * 2]
            sensor_4 = frame[unit * 3 + 60: unit * 4, frame_width // 3: frame_width // 3 * 2]

        elif self.current_direction == 'left':
            unit = frame_width // 4
            sensor_1 = frame[:unit - 60, frame_width // 3: frame_width // 3 * 2]
            sensor_2 = frame[unit: unit * 2, frame_width // 3: frame_width // 3 * 2]
            sensor_3 = frame[unit * 2: unit * 3, frame_width // 3: frame_width // 3 * 2]
            sensor_4 = frame[unit * 3 + 60: unit * 4, frame_width // 3: frame_width // 3 * 2]
                    
        elif self.current_direction == 'up':
            unit = frame_height // 4
            sensor_1 = frame[frame_height // 3: frame_height // 3 * 2, :unit - 60]
            sensor_2 = frame[frame_height // 3: frame_height // 3 * 2, unit: unit * 2]
            sensor_3 = frame[frame_height // 3: frame_height // 3 * 2, unit * 2: unit * 3]
            sensor_4 = frame[frame_height // 3: frame_height // 3 * 2, unit * 3 + 60 : unit * 4]

        elif self.current_direction == 'down':
            unit = frame_height // 4
            sensor_1 = frame[frame_height // 3: frame_height // 3 * 2, :unit - 60]
            sensor_2 = frame[frame_height // 3: frame_height // 3 * 2, unit: unit * 2]
            sensor_3 = frame[frame_height // 3: frame_height // 3 * 2, unit * 2: unit * 3]
            sensor_4 = frame[frame_height // 3: frame_height // 3 * 2, unit * 3 + 60: unit * 4]

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
        print(average_1, average_2, average_3, average_4)

        cv2.imshow("line_follower__frame", processed_frame)


def main():
    pid_controller_x.initialize()
    pid_controller_y.initialize()
    pid_controller_z.initialize()
    yaw_controller_pid.initialize()
    # Tello SDK
    drone = Tello()
    drone.connect()
    drone.streamon()
    frame_read = drone.get_frame_read()
    key = -1
    distance_to_target = 1e9

    # frame = frame_read.frame
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.threshold(gray_frame, THRESHOLD, 255, cv2.THRESH_BINARY)
    # blur_gray = cv2.GaussianBlur(gray_frame,(KERNEL_SIZE, KERNEL_SIZE), 0)
    # edges_frame = cv2.Canny(blur_gray, 128, 128)
    # dilated_frame = cv2.dilate(edges_frame, (KERNEL_SIZE, KERNEL_SIZE), iterations=5)

    # Aruco Tracking to specific range
    # while distance_to_target > 50:
    #     frame = frame_read.frame
    #     key = cv2.waitKey(33)
    #     cv2.imshow("frame", frame)

    #     tracking_result = tracking_aruco(frame, 3, 50)
    #     if key != -1:
    #         keyboard(drone, key)
    #     elif tracking_result:
    #         (
    #             labeled_frame,
    #             x_update,
    #             y_update,
    #             z_update,
    #             yaw_update,
    #             distance_to_target,
    #         ) = tracking_result
    #         print(distance_to_target)
    #         # send_drone_control(drone, key, x_update, y_update, z_update, yaw_update) 
    #         cv2.putText(
    #             labeled_frame,
    #             f"{x_update}, {y_update}, {z_update}, {yaw_update}",
    #             (200, 200),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.4,
    #             (0, 255, 0),
    #             2,
    #             cv2.LINE_AA,
    #         )
    #         cv2.imshow("aruco", labeled_frame)

    # Line Following
    line_follower = LineFollower(drone)
    while True:
        frame = drone.get_frame_read().frame
        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)
            continue
        else:
            line_follower.follow_line(frame)


if __name__ == "__main__":
    main()
