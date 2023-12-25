import cv2
import numpy as np
import time
import math
import keyboard_djitellopy
import random
from djitellopy import Tello
from pyimagesearch.pid import PID
from face_detect import FaceDetector
from lily_line_follow import LineFollow

# YOLOv7
import torch
from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, scale_coords
from utils.plots import plot_one_box


def estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    trash = []
    rvecs = []
    tvecs = []
    i = 0
    for c in corners:
        nada, R, t = cv2.solvePnP(
            marker_points, corners[i], mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash


class Drone:
    MAX_SPEED_THRESHOLD = 30
    SCALING_FACTOR = 1
    SCALING_FACTOR_H = 0.2
    SCALING_FACTOR_Y = 0.4
    SCALING_FACTOR_Z = 0.3
    HAND_CONTROL_FB = 20    # Front and back
    HAND_CONTROL_LR = 20    # Left and right
    HAND_CONTROL_UD = 35    # Up and down
    init = True
    time_cnt = 0
    wait_key_interval = 10
    interrupted_flag = False

    def __init__(self):
        self.fs = cv2.FileStorage("calibrate-01.xml", cv2.FILE_STORAGE_READ)
        self.intrinsic = self.fs.getNode("intrinsic").mat()
        self.distortion = self.fs.getNode("distortion").mat()

        # Face detector
        self.face_detector = FaceDetector()

        # Line follower
        self.line_follower = LineFollow()

        # Aruco marker
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # Tello and frame_read object
        self.drone = Tello()
        self.drone.connect()
        self.drone.streamon()
        self.frame_read = self.drone.get_frame_read()

        # PID Controller
        self.z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.h_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.yaw_pid.initialize()
        self.z_pid.initialize()
        self.h_pid.initialize()
        self.y_pid.initialize()

    def send_control(
        self,
        left_right_velocity,
        forward_backward_velocity,
        up_down_velocity,
        yaw_velocity,
    ):
        self.drone.send_rc_control(
            left_right_velocity,
            forward_backward_velocity,
            up_down_velocity,
            yaw_velocity,
        )

    def _clamping(self, val):
        if val > self.MAX_SPEED_THRESHOLD:
            return self.MAX_SPEED_THRESHOLD
        elif val < -self.MAX_SPEED_THRESHOLD:
            return -self.MAX_SPEED_THRESHOLD
        return int(val)

    def follow_line(self, cur_direction, next_direction) -> (np.ndarray, bool, int, int, int, int):
        """
        :param cur_direction: The current direction
        :param next_direction: The next direction
        :return: True if the feature is found and followed, False otherwise 
        """
        frame = self.frame_read.frame
        s = self.detect_line(self.frame_read.frame)  # sides = [top, bottom, left, right]
        drawn_frame = self.draw_sides(frame, s)
        print("[follow_line] sides: ", s)

        # put current direction on the frame
        cv2.putText(
            drawn_frame,
            f"cur_direction: {cur_direction}",
            (0, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        # put next direction on the frame
        cv2.putText(
            drawn_frame,
            f"next_direction: {next_direction}",
            (0, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.imshow("frame", frame)
        a = b = c = d = 0
        UP_SPD = 20
        DOWN_SPD = -30
        LEFT_SPD = -15
        RIGHT_SPD = 15
        BACK_SPD = -10

        # count the number of 1 in g
        count = 0
        for side in s:
            if side:
                count += 1
        if count == 0:
            # No black line detected, go back
            if cur_direction == "left":
                a = RIGHT_SPD
            elif cur_direction == "up":
                c = DOWN_SPD
            elif cur_direction == "right":
                a = LEFT_SPD
            elif cur_direction == "down":
                c = UP_SPD
            # b = BACK_SPD
            return False, a, b, c, d
        # if count > 5:  # Too close, go backward
        #     if not self.init:
        #         b = BACK_SPD

        if cur_direction == "left":
            a = LEFT_SPD
        elif cur_direction == "up":
            c = UP_SPD
        elif cur_direction == "right":
            a = RIGHT_SPD
        elif cur_direction == "down":
            c = DOWN_SPD

        # Check feature
        if cur_direction == "right" and next_direction == "up":  # 」↑
            if not s[3] or s[0] == 1: # right side is not black or top side is black
                return True, 0, 0, 0, 0
            elif s[0] == 2: # top is full black
                c = UP_SPD 
            elif s[1] == 2: # bottom is full black
                c = DOWN_SPD 

        elif cur_direction == "down" and next_direction == "left":  # 」←
            if not s[1] or s[2] == 1: # bottom side is not black or left side is black
                return True, 0, 0, 0, 0
            elif s[2] == 2: # left is full black
                a = LEFT_SPD
            elif s[3] == 2: # right is full black
                a = RIGHT_SPD

        elif cur_direction == "left" and next_direction == "down":  # 「 ↓
            if not s[2] or s[1] == 1: # left side is not black or bottom side is black
                return True, 0, 0, 0, 0
            elif s[0] == 2: # top is full black
                c = UP_SPD
            elif s[1] == 2: # bottom is full black
                c = DOWN_SPD

        elif cur_direction == "up" and next_direction == "right":  # 「 →
            if not s[0] or s[3] == 1: # top side is not black or right side is black
                return True, 0, 0, 0, 0
            elif s[2] == 2: # left is full black:
                a = LEFT_SPD
            elif s[3] == 2: # right is full black
                a = RIGHT_SPD

        elif cur_direction == "up" and next_direction == "left":  # 7 ←
            if not s[0] or s[2] == 1: # top side is not black or left side is black
                return True, 0, 0, 0, 0
            elif s[2] == 2: # left is full black
                a = LEFT_SPD
            elif s[3] == 2: # right is full black
                a = RIGHT_SPD

        elif cur_direction == "left" and next_direction == "up":  # L ↑
            if not s[2] or s[0] == 1: # left side is not black or top side is black
                return True, 0, 0, 0, 0
            elif s[0] == 2: # top is full black
                c = UP_SPD
            elif s[1] == 2: # bottom is full black
                c = DOWN_SPD

        return False, a, b, c, d


    def follow_marker(self, marker_id, distance) -> (np.ndarray, bool, int, int, int, int):
        """
        :param marker_id: The id of the marker to follow
        :param distance: The distance to keep from the marker
        :return: True if the marker if found and reached, False otherwise
        """
        frame = self.frame_read.frame
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            frame, self.dictionary, parameters=self.parameters
        )
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            markerCorners, 15, self.intrinsic, self.distortion
        )
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        if np.array(rvec).ndim == 0:
            return frame, False, 0, 0, 0, 0

        frame_width, frame_height = frame.shape[1], frame.shape[0]

        for i in range(len(markerIds)):
            if markerIds[i] == marker_id:
                center_x = (markerCorners[i][0][0]
                            [0] + markerCorners[i][0][2][0]) / 2
                center_y = (markerCorners[i][0][0]
                            [1] + markerCorners[i][0][2][1]) / 2
                horizontal_offset = center_x - frame_width / 2
                vertical_offset = -1 * (center_y - frame_height / 2)

                horizontal_update = horizontal_offset * self.SCALING_FACTOR_H
                vertical_update = (vertical_offset - 50) * \
                    self.SCALING_FACTOR_Y

                rot_mat, _ = cv2.Rodrigues(rvec[i])
                euler_angles = cv2.RQDecomp3x3(rot_mat)
                yaw_angle = euler_angles[0][2]
                yaw_control = yaw_angle * self.SCALING_FACTOR

                z_update = tvec[i, 0, 2] - distance
                if (
                    z_update < 15
                    and z_update > -15
                    and yaw_angle < 15
                    and yaw_angle > -15
                ):
                    return (
                        frame,
                        True,
                        int(horizontal_update),
                        int(z_update // 2),
                        int(vertical_update // 2),
                        int(yaw_control),
                    )

                z_update = self._clamping(self.z_pid.update(z_update, sleep=0))
                horizontal_update = self._clamping(
                    self.h_pid.update(horizontal_update, sleep=0))
                vertical_update = self._clamping(
                    self.y_pid.update(vertical_update, sleep=0))

                text = f"[follow_marker] Target marker detected h : {horizontal_update} f : {z_update} v : {vertical_update} r : {yaw_control}"
                cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                return (
                    frame,
                    False,
                    int(horizontal_update),
                    int(z_update // 2),
                    int(vertical_update // 2),
                    int(yaw_control),
                )

        print("[follow_marker] No marker detected, stay still!")
        return frame, False, 0, 0, 0, 0

    def follow_face(self) -> (bool, int, int, int, int):
        frame = self.frame_read.frame
        frame, dist = self.face_detector.detect(frame)
        if dist is None:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame, False, 0, 0, 0, 0
        if dist > self.face_detector.threshold:
            return frame, False, 0, 0, 0, 5
        if dist < self.face_detector.threshold:
            return frame, False, 0, 0, 0, -5
        return frame, True, 0, 0, 0, 0

    def find_marker(self) -> list[int]:
        """
        :return: A list of marker ids found in the frame
        """
        frame = self.frame_read.frame
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            frame, self.dictionary, parameters=self.parameters
        )
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        return frame, markerIds

    def command_loop(
        self,
        command="stop",
        marker_id=0,
        direction=0,
        distance=0,
        next_direction=0,
        valid_count=1,
        height=0,
    ):
        """
        This part control the drone and handle the keyboard input at the same time
        """
        tmp_valid_count = valid_count
        while True:
            frame = self.frame_read.frame
            key = cv2.waitKey(self.wait_key_interval)
            if self.init == True:
                self.time_cnt += self.wait_key_interval
                if self.time_cnt > 2000:
                    self.init = False
                    self.time_cnt = 0

            # DONT WRITE LOOPS INSIDE THIS CONDITION CHAIN
            if key != -1:
                print(key)
                self.interrupted_flag = True
                if key == ord("p"):
                    self.interrupted_flag = not self.interrupted_flag
                keyboard_djitellopy.keyboard(self.drone, key)
            elif command == "take_off":
                self.drone.takeoff()
                return
            elif command == "land":
                self.drone.land()
                return
            elif command == "turn_right":
                self.drone.rotate_clockwise(90)
                return
            elif command == "turn_back":
                self.drone.rotate_clockwise(180)
                return
            elif command == "stop":
                self.send_control(0, 0, 0, 0)
                return
            elif command == "forward":
                valid_count -= 1
                cv2.putText(frame, f"valid_count: {valid_count}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if valid_count == 0:
                    self.send_control(0, 0, 0, 0)
                    return
                self.send_control(0, self.HAND_CONTROL_FB, 0, 0)
            elif command == "up_until_marker":
                # self.send_control(0, 0, 10, 0)
                frame, marker_ids = self.find_marker()
                if marker_ids:
                    self.send_control(0, 0, 0, 0)
                    return
            elif command == "direct_move_left":
                if self.interrupted_flag:
                    self.send_control(0, 0, 0, 0)
                    valid_count = -1
                else:
                    valid_count -= 1
                    cv2.putText(frame, f"valid_count: {valid_count}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if valid_count < 0:
                        self.send_control(0, 0, 0, 0)
                        return
                    self.send_control(-self.HAND_CONTROL_LR, 0, 0, 0)
            elif command == "direct_move_right":
                if self.interrupted_flag:
                    self.send_control(0, 0, 0, 0)
                    valid_count = -1
                else:
                    valid_count -= 1
                    cv2.putText(frame, f"valid_count: {valid_count}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if valid_count < 0:
                        self.send_control(0, 0, 0, 0)
                        return
                    self.send_control(self.HAND_CONTROL_LR, 0, 0, 0)
            elif command == "direct_move_up":
                if self.interrupted_flag:
                    self.send_control(0, 0, 0, 0)
                else:
                    my_height = self.drone.get_height()
                    cv2.putText(frame, f"my_height: {my_height}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if my_height > height:
                        self.send_control(0, 0, 0, 0)
                        return
                    self.send_control(0, 0, self.HAND_CONTROL_UD, 0)
            elif command == "direct_move_down":
                if self.interrupted_flag:
                    self.send_control(0, 0, 0, 0)
                else:
                    my_height = self.drone.get_height()
                    cv2.putText(frame, f"my_height: {my_height}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if my_height < height:
                        self.send_control(0, 0, 0, 0)
                        return
                    self.send_control(0, 0, -self.HAND_CONTROL_UD, 0)
            elif command == "follow_marker":
                frame, is_success, x, z, y, yaw = self.follow_marker(
                    marker_id, distance)
                # Put x, z, y, yaw on the frame
                cv2.putText(frame, f"x: {x}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"z: {z}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"y: {y}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"yaw: {yaw}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if is_success:
                    print("Success follow marker!!!!!")
                    self.send_control(0, 0, 0, 0)
                    return
                self.send_control(x, z, y, yaw)
            elif command == "follow_face":
                frame, is_success, x, z, y, yaw = self.follow_face()

                # Put x, z, y, yaw on the frame
                if is_success:
                    valid_count -= 1
                    self.send_control(0, 0, 0, 0)
                    cv2.putText(frame, f"valid_count: {valid_count}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if valid_count == 0:
                        return
                else:
                    valid_count = tmp_valid_count
                cv2.putText(frame, f"x: {x}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"z: {z}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"y: {y}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"yaw: {yaw}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.send_control(x, z, y, yaw)
            elif command == "follow_line":
                frame, is_success, x, z, y, yaw = self.follow_line(
                    direction, next_direction)

                if is_success:
                    valid_count -= 1
                    self.send_control(0, 0, 0, 0)
                    cv2.putText(frame, f"valid_count: {valid_count}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if valid_count == 0:
                        return
                else:
                    valid_count = tmp_valid_count
                # Put x, z, y, yaw on the frame
                cv2.putText(frame, f"x: {x}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"z: {z}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"y: {y}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"yaw: {yaw}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                self.send_control(x, z, y, yaw)
            else:
                print("[handle_command] Invalid command:", command)

            cv2.imshow("frame", frame)


def main():
    drone = Drone()

    # 0:left, 1:up, 2:right, 3:down
    # line ['line', [0123], [line_position]] 離牆30
    actions_test = [
        {"command": "take_off"},
        {"command": "up_until_marker"},
        {"command": "follow_marker", "marker_id": 0, "distance": 50},
        
        {"command": "stop"},
        {"command": "land"},
    ]

    actions_lab11 = [
        {"command": "take_off"},
        {"command": "up_until_marker"},
        {"command": "follow_marker", "marker_id": 3, "distance": 36},
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "up",
        },
        {
            "command": "follow_line",
            "direction": "up",
            "next_direction": "left",
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
        },
        {
            "command": "follow_line",
            "direction": "down",
            "next_direction": "left",
        },
        {
            "command": "follow_line",
            "direction": "left",
            "next_direction": "down",
        },
        {"command": "land"},
    ]

    active_actions = actions_lab11

    # YOLOv7 detech object to decide which action to take
    # WEIGHT = './best.pt'
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = attempt_load(WEIGHT, map_location=device)
    # if device == "cuda":
    #     model = model.half().to(device)
    # else:
    #     model = model.float().to(device)
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    # while len(active_actions) == 0:
    #     image = drone.frame_read.frame
    #     image_labeled = image.copy()

    #     image = letterbox(image, (640, 640), stride=64, auto=True)[0]
    #     if device == "cuda":
    #         image = transforms.ToTensor()(image).to(device).half().unsqueeze(0)
    #     else:
    #         image = transforms.ToTensor()(image).to(device).float().unsqueeze(0)

    #     with torch.no_grad():
    #         output = model(image)[0]
    #     output = non_max_suppression_kpt(output, conf_thres=0.25, iou_thres=0.65)[0]

    #     output[:, :4] = scale_coords(image.shape[2:], output[:, :4], image_labeled.shape).round()
    #     for *xyxy, conf, cls in output:
    #         label = f'{names[int(cls)]} {conf:.2f}'
    #         if names[int(cls)] == 'carna':
    #             active_actions = actions_carna
    #             break
    #         elif names[int(cls)] == 'melody':
    #             active_actions = actions_melody
    #             break
    #         plot_one_box(xyxy, image_labeled, label=label, color=colors[int(cls)], line_thickness=1)

    #     cv2.waitKey(3)
    #     cv2.imshow("YOLOv7", image_labeled)
    #     cv2.destroyWindow("YOLOv7")

    for action in active_actions:
        print("[Main loop] current_action:", action)
        drone.command_loop(**action)


if __name__ == "__main__":
    main()
