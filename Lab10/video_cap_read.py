import numpy as np
import cv2
import os
from djitellopy import Tello
from pyimagesearch.pid import PID
from keyboard_djitellopy import keyboard


def main():
   video = cv2.VideoCapture('./output.avi')
   
   while True:
       ret, frame = video.read()
       
       if not ret:
           break
       
       cv2.imshow('frame', frame)
       cv2.waitKey(30)
       
main()