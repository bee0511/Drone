import cv2
import numpy as np
import time
import math
import keyboard_djitellopy
from djitellopy import Tello
from pyimagesearch.pid import PID

def keyboard(self, key):
    #global is_flying
    print("key:", key)
    fb_speed = 40
    lf_speed = 40
    ud_speed = 50
    degree = 30
    if key == ord('1'):
        self.takeoff()
        #is_flying = True
    if key == ord('2'):
        self.land()
        #is_flying = False
    if key == ord('3'):
        self.send_rc_control(0, 0, 0, 0)
        print("stop!!!!")
    if key == ord('w'):
        self.send_rc_control(0, fb_speed, 0, 0)
        print("forward!!!!")
    if key == ord('s'):
        self.send_rc_control(0, (-1) * fb_speed, 0, 0)
        print("backward!!!!")
    if key == ord('a'):
        self.send_rc_control((-1) * lf_speed, 0, 0, 0)
        print("left!!!!")
    if key == ord('d'):
        self.send_rc_control(lf_speed, 0, 0, 0)
        print("right!!!!")
    if key == ord('z'):
        self.send_rc_control(0, 0, ud_speed, 0)
        print("down!!!!")
    if key == ord('x'):
        self.send_rc_control(0, 0, (-1) *ud_speed, 0)
        print("up!!!!")
    if key == ord('c'):
        self.send_rc_control(0, 0, 0, degree)
        print("rotate!!!!")
    if key == ord('v'):
        self.send_rc_control(0, 0, 0, (-1) *degree)
        print("counter rotate!!!!")
    if key == ord('5'):
        height = self.get_height()
        print(height)
    if key == ord('6'):
        battery = self.get_battery()
        print (battery)

class Drone:
    def __init__(self):
        self.max_speed_threadhold = 30
        self.scaling_factor = 1.2
        self.scaling_factor_h = 0.2
        self.scaling_factor_y = 0.4
        self.scaling_factor_z = 0.3
        
        self.fs = cv2.FileStorage("calibration_result.xml", cv2.FILE_STORAGE_READ)
        self.intrinsic = self.fs.getNode("intrinsic").mat()
        self.distortion = self.fs.getNode("distortion").mat()
        # Load the predefined dictionary
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # Initialize the detector parameters using default values
        self.parameters = cv2.aruco.DetectorParameters_create()
        
        # Tello
        self.drone = Tello()
        self.drone.connect()
        
        self.frame = 0
        self.line_exp = 0
        self.line_now = 0
        
        self.z_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.h_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.y_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        self.yaw_pid = PID(kP=0.7, kI=0.0001, kD=0.1)
        
        self.yaw_pid.initialize()
        self.z_pid.initialize()
        self.h_pid.initialize()
        self.y_pid.initialize()
        
        
        self.drone.streamon()
        self.frame_read = self.drone.get_frame_read()

        # 0:left, 1:up, 2:right, 3:down
        #line ["line", [0123], [line_position]] 離牆30
        self.action_list = [["take_off"], ["up_u", 2], ["find_marker", 2, 30], ["left", 0.5], 
                            ["line", 0, [0, 1, 0, 1, 1, 1, 0, 0, 0]], #left
                            ["line", 1, [0, 0, 0, 1, 1, 0, 0, 1, 0]], #up 
                            ["line", 0, [0, 0, 0, 0, 1, 1, 0, 1, 0]], #left
                            ["line", 3, [0, 1, 0, 1, 1, 1, 0, 0, 0]], #down
                            ["left", 3], 
                            ["land"]]

    def calculate_black_area(self, cell):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for black color in HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])

        # Threshold the HSV image to get only black colors
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Calculate the area of black regions
        black_area = np.sum(mask == 255)

        return black_area

    def detect_line_in_grid(self):
        height, width = self.frame.shape[:2]
        cells = [
            self.frame[i * height // 3: (i + 1) * height // 3, j * width // 3: (j + 1) * width // 3]
            for i in range(3) for j in range(3)
        ]

        line_positions = []

        for cell in cells:
            line_position = self.calculate_black_area(cell)  # Use your line detection method here
            if line_position > 17000:
                line_positions.append(1)
            else:
                line_positions.append(0)
        
        return line_positions

    def draw_grid(self, frame):
        height, width, _ = frame.shape

        # Draw vertical lines to divide the frame into 3 columns
        cv2.line(frame, (width // 3, 0), (width // 3, height), (255, 0, 0), 2)
        cv2.line(frame, (2 * width // 3, 0), (2 * width // 3, height), (255, 0, 0), 2)

        # Draw horizontal lines to divide the frame into 3 rows
        cv2.line(frame, (0, height // 3), (width, height // 3), (255, 0, 0), 2)
        cv2.line(frame, (0, 2 * height // 3), (width, 2 * height // 3), (255, 0, 0), 2)

        return frame
    
    def determine_action(self, action_now):
        if self.action_list[action_now][1] == 0:
            a, b, c, d = -10, 0, 0, 0
        elif self.action_list[action_now][1] == 1:
            a, b, c, d = 0, 0, 15, 0
        elif self.action_list[action_now][1] == 2:
            a, b, c, d = 10, 0, 0, 0
        else:
            a, b, c, d = 0, 0, -15, 0
        self.line_exp = self.action_list[action_now][2]
        self.line_now = self.detect_line_in_grid()
        print("line_exp:", self.line_exp)
        print("line_now:", self.line_now)
        if self.line_exp == [0, 0, 0, 0, 1, 1, 0, 1, 0]:     # 『
            if self.line_now == [0, 1, 1, 0, 1, 0, 0, 1, 0]:
                a, b, c, d = 0, 0, 10, 0
            elif self.line_now == [0, 0, 0, 0, 0, 0, 0, 1, 1]:
                a, b, c, d = 0, 0, -10, 0
            elif self.line_now == [0, 1, 0, 0, 1, 0, 0, 1, 0]:
                a, b, c, d = 0, 0, 10, 0
            elif self.line_now == [0, 0, 0, 0, 0, 1, 0, 0, 1]:
                a, b, c, d == 10, 0, 0, 0
            elif self.line_now == [0, 0, 0, 0, 1, 1, 0, 0, 0]:
                a, b, c, d == 0, 5, 0, 0
        elif self.line_exp == [0, 1, 0, 1, 1, 0, 0, 0, 0]:   # 』
            if self.line_now == [0, 1, 0, 0, 1, 0, 1, 1, 0]:
                a, b, c, d = 0, 0, -10, 0
            elif self.line_now == [1, 1, 0, 0, 0, 0, 0, 0, 0]:
                a, b, c, d = 0, 0, 10, 0
            elif self.line_now == [1, 0, 0, 1, 0, 0, 0, 0, 0]:
                a, b, c, d = -10, 0, 0, 0
            elif self.line_now == [0, 1, 0, 0, 1, 0, 0, 1, 0]:
                a, b, c, d = 0, 0, -10, 0
        elif self.line_exp == [0, 0, 0, 1, 1, 0, 0, 1, 0]:   #7
            if self.line_now == [0, 0, 0, 1, 0, 0, 1, 0, 0]:
                a, b, c, d = -10, 0, 0, 0
            elif self.line_now == [0, 0, 0, 1, 1, 1, 0, 0, 1]:
                a, b, c, d = 10, 0, 0, 0
        elif self.line_exp == [0, 1, 0, 1, 1, 1, 0, 0, 0]:
            if self.line_now == [1, 1, 1, 0, 0, 0, 0, 0, 0]:
                a, b, c, d = 0, 0, 10, 0
            elif self.line_now == [0, 1, 0, 0, 1, 0, 1, 1, 1]:
                a, b, c, d = 0, 0, -10, 0
            elif self.line_now == [1, 0, 0, 1, 0, 0, 1, 1, 1]:
                a, b, c, d = -10, 0, -10, 0
            elif self.line_now == [0, 0, 1, 0, 0, 1, 1, 1, 1]:
                a, b, c, d = 10, 0, -10, 0
            elif self.line_now == [0, 0, 1, 1, 1, 1, 0, 0, 0]:
                a, b, c, d = 10, 0, 0, 0
            elif self.line_now == [1, 0, 0, 1, 1, 1, 0, 0, 0]:
                a, b, c, d = -10, 0, 0, 0

        return a, b, c, d
    
    def find_marker(self):
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(frame, self.dictionary, parameters=self.parameters)
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        
        if len(markerCorners) > 0:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, self.intrinsic, self.distortion)
            frame_width, frame_height = frame.shape[1], frame.shape[0]
            
            for i in range(len(markerIds)):
                if markerIds[i] == self.action_list[action_now][1]:
                    
                    center_x = (markerCorners[i][0][0][0] + markerCorners[i][0][2][0]) / 2
                    center_y = (markerCorners[i][0][0][1] + markerCorners[i][0][2][1]) / 2
                    horizontal_offset = center_x - frame_width / 2
                    vertical_offset = -1 * (center_y - frame_height / 2)
                    
                    horizontal_update = horizontal_offset * self.scaling_factor_h
                    vertical_update = vertical_offset * self.scaling_factor_y
            
            
                    rot_mat, _ = cv2.Rodrigues(rvec[i])
                    euler_angles = cv2.RQDecomp3x3(rot_mat)
                    yaw_angle = euler_angles[0][2]
                    yaw_control = yaw_angle * self.scaling_factor
        
                    z_update = tvec[i, 0, 2] - self.action_list[action_now][2]
                    if z_update < 15 and z_update > -15 and yaw_angle < 15 and yaw_angle > -15:
                        action_now = action_now + 1
                    
                    z_update = self.z_pid.update(z_update, sleep=0)
                    
                    horizontal_update = self.h_pid.update(horizontal_update, sleep=0)
                    vertical_update = self.y_pid.update(vertical_update, sleep=0)
                    
                    if horizontal_update > self.max_speed_threadhold:
                        horizontal_update = self.max_speed_threadhold
                    elif horizontal_update < -self.max_speed_threadhold:
                        horizontal_update = -self.max_speed_threadhold
                    
                    if vertical_update > self.max_speed_threadhold:
                        vertical_update = self.max_speed_threadhold
                    elif vertical_update < -self.max_speed_threadhold:
                        vertical_update = -self.max_speed_threadhold
                    
                    if z_update > self.max_speed_threadhold:
                        z_update = self.max_speed_threadhold
                    elif z_update < -self.max_speed_threadhold:
                        z_update = -self.max_speed_threadhold
                        
                    print("h : ",  horizontal_update, "f : ", z_update, "v : ", vertical_update, "r : ", yaw_control)
                    self.send_rc_control(int(horizontal_update), int(z_update // 2), int(vertical_update // 2), int(yaw_control))
        else :
            self.send_rc_control(0, 0, 0, 0)
    
    def find_marker_u(self):
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(self.frame, self.dictionary, parameters=self.parameters)
        frame = cv2.aruco.drawDetectedMarkers(self.frame, markerCorners, markerIds)
        
        if len(markerCorners) > 0:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 15, self.intrinsic, self.distortion)
            frame_width, frame_height = frame.shape[1], frame.shape[0]
            
            for i in range(len(markerIds)):
                if markerIds[i] == self.action_list[action_now][1]:
                    
                    center_x = (markerCorners[i][0][0][0] + markerCorners[i][0][2][0]) / 2
                    center_y = (markerCorners[i][0][0][1] + markerCorners[i][0][2][1]) / 2
                    horizontal_offset = center_x - frame_width / 2
                    vertical_offset = -1 * (center_y - frame_height / 2)
                    
                    horizontal_update = horizontal_offset * self.scaling_factor_h
                    vertical_update = vertical_offset * self.scaling_factor_y
            
            
                    rot_mat, _ = cv2.Rodrigues(rvec[i])
                    euler_angles = cv2.RQDecomp3x3(rot_mat)
                    yaw_angle = euler_angles[0][2]
                    yaw_control = yaw_angle * self.scaling_factor
        
                    z_update = tvec[i, 0, 2] - self.action_list[action_now][2]
                    
                    z_update = self.z_pid.update(z_update, sleep=0)
                    
                    horizontal_update = self.h_pid.update(horizontal_update, sleep=0)
                    vertical_update = self.y_pid.update(vertical_update, sleep=0)
                    
                    if horizontal_update > self.max_speed_threadhold:
                        horizontal_update = self.max_speed_threadhold
                    elif horizontal_update < -self.max_speed_threadhold:
                        horizontal_update = -self.max_speed_threadhold
                    
                    if vertical_update > self.max_speed_threadhold:
                        vertical_update = self.max_speed_threadhold
                    elif vertical_update < -self.max_speed_threadhold:
                        vertical_update = -self.max_speed_threadhold
                    
                    if z_update > self.max_speed_threadhold:
                        z_update = self.max_speed_threadhold
                    elif z_update < -self.max_speed_threadhold:
                        z_update = -self.max_speed_threadhold
                        
                    print("h : ",  horizontal_update, "f : ", z_update, "v : ", vertical_update, "r : ", yaw_control)
                    self.drone.send_rc_control(int(horizontal_update), int(z_update // 2), int(vertical_update // 2), int(yaw_control))
                elif markerIds[i] == self.action_list[action_now][3]:
                    
                    center_x = (markerCorners[i][0][0][0] + markerCorners[i][0][2][0]) / 2
                    center_y = (markerCorners[i][0][0][1] + markerCorners[i][0][2][1]) / 2
                    horizontal_offset = center_x - frame_width / 2
                    vertical_offset = -1 * (center_y - frame_height / 2)
                    
                    horizontal_update = horizontal_offset * self.scaling_factor_h
                    vertical_update = vertical_offset * self.scaling_factor_y
            
            
                    rot_mat, _ = cv2.Rodrigues(rvec[i])
                    euler_angles = cv2.RQDecomp3x3(rot_mat)
                    yaw_angle = euler_angles[0][2]
                    yaw_control = yaw_angle * self.scaling_factor
        
                    z_update = tvec[i, 0, 2] - self.action_list[action_now][4]
                    if z_update < 15 and z_update > -15 and yaw_angle < 15 and yaw_angle > -15:
                        self.drone.send_rc_control(0, 0, 0, 0)
                        action_now = action_now + 1
                        break
                    
                    z_update = self.z_pid.update(z_update, sleep=0)
                    
                    horizontal_update = self.h_pid.update(horizontal_update, sleep=0)
                    vertical_update = self.y_pid.update(vertical_update, sleep=0)
                    
                    if horizontal_update > self.max_speed_threadhold:
                        horizontal_update = self.max_speed_threadhold
                    elif horizontal_update < -self.max_speed_threadhold:
                        horizontal_update = -self.max_speed_threadhold
                    
                    if vertical_update > self.max_speed_threadhold:
                        vertical_update = self.max_speed_threadhold
                    elif vertical_update < -self.max_speed_threadhold:
                        vertical_update = -self.max_speed_threadhold
                    
                    if z_update > self.max_speed_threadhold:
                        z_update = self.max_speed_threadhold
                    elif z_update < -self.max_speed_threadhold:
                        z_update = -self.max_speed_threadhold
                        
                    print("h : ",  horizontal_update, "f : ", z_update, "v : ", vertical_update, "r : ", yaw_control)
                    self.drone.send_rc_control(int(horizontal_update), int(z_update // 2), int(vertical_update // 2), int(yaw_control))
        else :
            self.drone.send_rc_control(0, 0, 0, 0)

    def find_face(self):
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(self.frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        frame_width, frame_height = self.frame.shape[1], self.frame.shape[0]

        two_face = []
        for face in faces:
            x = face[0]
            y = face[1]
            w = face[2]
            h = face[3]
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            black_line_wid = 1/4*h
            p1 = (int(x-1/2*w), int(y-black_line_wid-h/2))
            p2 = (int(x+3/2*w), int(y-h/2))
            p3 = (int(x-1/2*w), int(y+h/2+h))
            p4 = (int(x+3/2*w), int(y+h/2+black_line_wid+h))
            black_space = [(p1, p2), (p3, p4)]
            cv2.rectangle(self.frame, black_space[0][0], black_space[0][1], (0,0,0), 2)
            cv2.rectangle(self.frame, black_space[1][0], black_space[1][1], (0,0,0), 2)
            roi1 = self.frame[black_space[0][0][0]:black_space[0][1][0], black_space[0][0][1]:black_space[0][1][1]]
            roi2 = self.frame[black_space[1][0][0]:black_space[1][1][0], black_space[1][0][1]:black_space[1][1][1]]
            avg_c1 = roi1.mean(axis=(0,1))
            avg_c2 = roi2.mean(axis=(0,1))
            
            threshold = 120
            c1_is_black = False
            if all(avg_c1 < threshold):
                cv2.putText(self.frame, f"is Black", black_space[0][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                c1_is_black = True
            else:
                cv2.putText(self.frame, f"NOT Black!!!", black_space[0][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            c2_is_black = False
            if all(avg_c2 < threshold):
                cv2.putText(self.frame, f"is Black", black_space[1][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                c2_is_black = True
            else:
                cv2.putText(self.frame, f"NOT Black!!!", black_space[1][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            focal_length_mm = 75000
            if c1_is_black and c2_is_black:
                face_size_pixels = w * h
                distance_cm = (focal_length_mm * 15) / face_size_pixels
                two_face.append((x, y, w, h, distance_cm))
                cv2.putText(self.frame, "this pic in two face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        if len(two_face) == 2:
            center_x = ((two_face[0][0] + two_face[0][2]/2) + (two_face[1][0] + two_face[1][2]/2)) / 2
            center_y = ((two_face[0][1] + two_face[0][3]/2) + (two_face[1][1] + two_face[1][3]/2)) / 2
            horizontal_offset = center_x - frame_width / 2
            vertical_offset = -1 * (center_y - frame_height / 2)
            dist = 30
            z_offset = dist - (two_face[0][4] + two_face[1][4])/2
            
            if horizontal_offset < 3 and vertical_offset < 3:
                yaw_offset = two_face[0][2]*two_face[0][3] - two_face[1][2]*two_face[1][3]
            else:
                yaw_offset = 0
            
            horizontal_update = horizontal_offset * self.scaling_factor_h
            vertical_update = vertical_offset * self.scaling_factor_y
            z_update = z_offset * self.scaling_factor_z
            yaw_update = yaw_offset * self.scaling_factor 
            
            yaw_update = self.yaw_pid.update(yaw_update, sleep=0)
            z_update = self.z_pid.update(z_update, sleep=0)
            horizontal_update = self.h_pid.update(horizontal_update, sleep=0)
            vertical_update = self.y_pid.update(vertical_update, sleep=0)
            
            if horizontal_update > self.max_speed_threadhold:
                horizontal_update = self.max_speed_threadhold
            elif horizontal_update < -self.max_speed_threadhold:
                horizontal_update = -self.max_speed_threadhold
            
            if vertical_update > self.max_speed_threadhold:
                vertical_update = self.max_speed_threadhold
            elif vertical_update < -self.max_speed_threadhold:
                vertical_update = -self.max_speed_threadhold
            
            if z_update > self.max_speed_threadhold:
                z_update = self.max_speed_threadhold
            elif z_update < -self.max_speed_threadhold:
                z_update = -self.max_speed_threadhold
                
            print("h : ",  horizontal_update, "f : ", z_update, "v : ", vertical_update, "r : ", yaw_update)
            self.drone.send_rc_control(int(horizontal_update), int(z_update // 2), int(vertical_update // 2), int(yaw_update))
    

def main():
    drone = Drone()
    
    # 0:left, 1:up, 2:right, 3:down
    #line ["line", [0123], [line_position]] 離牆30
    action_now = 0 
    while action_now < len(drone.action_list):
        
        drone.frame = drone.frame_read.frame
        
        #看action_list
        #確定是否完成
        #是 下一個action
        #否 做
        if drone.action_list[action_now][0] == "line":    
            a, b, c, d = drone.determine_action(action_now=action_now)        
            #check too close or far
            s = 0
            for i in drone.line_now:
                if i == 1:
                    s += 1
            if s >= 6:
                a, b, c, d = 0, -5, 0, 0
            elif s <= 1:
                a, b, c, d = 0, 5, 0, 0                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            
            if drone.line_now == drone.line_exp:
                action_now = action_now + 1
            else:
                drone.send_rc_control(a, b, c, d)
        
        elif drone.action_list[action_now][0] == "stop":
            drone.send_rc_control(0, 0, 0, 0)
            time.sleep(drone.action_list[action_now][1])
            action_now = action_now + 1
        elif drone.action_list[action_now][0] == "take_off":
            drone.takeoff()
            action_now = action_now + 1
        elif drone.action_list[action_now][0] == "land":
            drone.land()
            action_now = action_now + 1
        elif drone.action_list[action_now][0] == "front":
            drone.send_rc_control(0, 30, 0, 0)
            time.sleep(drone.action_list[action_now][1])
            drone.send_rc_control(0, 0, 0, 0)
            action_now = action_now + 1
        elif drone.action_list[action_now][0] == "back":
            drone.send_rc_control(0, -30, 0, 0)
            time.sleep(drone.action_list[action_now][1])
            drone.send_rc_control(0, 0, 0, 0)
            action_now = action_now + 1
        elif drone.action_list[action_now][0] == "up":
            print("---up---")
            drone.send_rc_control(0, 0, 40, 0)
            time.sleep(drone.action_list[action_now][1])
            drone.send_rc_control(0, 0, 0, 0)
            action_now = action_now + 1
        elif drone.action_list[action_now][0] == "down":
            drone.send_rc_control(0, 0, -40, 0)
            time.sleep(drone.action_list[action_now][1])
            drone.send_rc_control(0, 0, 0, 0)
            action_now = action_now + 1
        elif drone.action_list[action_now][0] == "left":
            drone.send_rc_control(-30, 0, 0, 0)
            time.sleep(drone.action_list[action_now][1])
            drone.send_rc_control(0, 0, 0, 0)
            action_now = action_now + 1
        elif drone.action_list[action_now][0] == "rotate":
            drone.send_rc_control(0, 0, 0, -50)
            time.sleep(drone.action_list[action_now][1])
            drone.send_rc_control(0, 0, 0, 0)
            action_now = action_now + 1
        elif drone.action_list[action_now][0] == "up_u":
            drone.send_rc_control(0, 0, 30, 0)
            markerCorners, markerIds, _ = cv2.aruco.detectMarkers(frame, drone.dictionary, parameters=drone.parameters)
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            if len(markerCorners) > 0:
                for i in markerIds:
                    if i == drone.action_list[action_now][1]:
                        drone.send_rc_control(0, 0, 0, 0)
                        print(f"---ID: {drone.action_list[action_now][1]} is found!---")
                        action_now = action_now + 1
        elif drone.action_list[action_now][0] == "down_u":
            drone.send_rc_control(0, 0, -30, 0)
            markerCorners, markerIds, _ = cv2.aruco.detectMarkers(frame, drone.dictionary, parameters=drone.parameters)
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            if len(markerCorners) > 0:
                for i in markerIds:
                    if i == drone.action_list[action_now][1]:
                        drone.send_rc_control(0, 0, 0, 0)
                        print(f"---ID: {drone.action_list[action_now][1]} is found!---")
                        action_now = action_now + 1
        elif drone.action_list[action_now][0] == "left_u":
            drone.send_rc_control(-30, 0, 0, 0)
            markerCorners, markerIds, _ = cv2.aruco.detectMarkers(frame, drone.dictionary, parameters=drone.parameters)
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            if len(markerCorners) > 0:
                for i in markerIds:
                    if i == drone.action_list[action_now][1]:
                        drone.send_rc_control(0, 0, 0, 0)
                        print(f"---ID: {drone.action_list[action_now][1]} is found!---")
                        action_now = action_now + 1
        elif drone.action_list[action_now][0] == "rotate_u":
            drone.send_rc_control(0, 0, 0, 30)
            markerCorners, markerIds, _ = cv2.aruco.detectMarkers(frame, drone.dictionary, parameters=drone.parameters)
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            if len(markerCorners) > 0:
                for i in markerIds:
                    if i == drone.action_list[action_now][1]:
                        drone.send_rc_control(0, 0, 0, 0)
                        print(f"---ID: {drone.action_list[action_now][1]} is found!---")
                        action_now = action_now + 1
        elif drone.action_list[action_now][0] == "find_marker":
            drone.find_marker()
        
        elif drone.action_list[action_now][0] == "find_marker_u":
            drone.find_marker_u()
                
        elif drone.action_list[action_now][0] == "find_two_face":
            drone.find_face()

        cv2.imshow("f", frame)
        key = cv2.waitKey(33)
        if key != -1:
            keyboard(drone, key)
        if key == 27:
            break
        

if __name__ == '__main__':
    main()
        
