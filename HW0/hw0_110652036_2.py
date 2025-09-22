#HW0-2
import cv2
import numpy as np

video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

backSub = cv2.createBackgroundSubtractorMOG2()

for number in range(2):
    ret, frame = cap.read()
    fgMask = backSub.apply(frame)
    filename = 'hw0_110652036_2.png'
    cv2.imwrite(filename, fgMask)

cap.release()
cv2.destroyAllWindows()
