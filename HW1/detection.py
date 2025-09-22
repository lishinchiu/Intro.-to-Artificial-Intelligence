import os
import re
from turtle import Turtle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
from os import walk
from os.path import join
from datetime import datetime


def crop(x1, y1, x2, y2, x3, y3, x4, y4, img) :
    """
    This function ouput the specified area (parking space image) of the input frame acobjecting to the input of four xy coordinates.
    
      Parameters:
        (x1, y1, x2, y2, x3, y3, x4, y4, frame)
        
        (x1, y1) is the lower left corner of the specified area
        (x2, y2) is the lower right corner of the specified area
        (x3, y3) is the upper left corner of the specified area
        (x4, y4) is the upper right corner of the specified area
        frame is the frame you want to get it's parking space image
        
      Returns:
        parking_space_image (image size = 360 x 160)
      
      Usage:
        parking_space_image = crop(x1, y1, x2, y2, x3, y3, x4, y4, img)
    """
    lu = (x1, y1)
    ld = (x3, y3)
    ru = (x2, y2)
    rd = (x4, y4)
    
    
    dst = np.array([[0, 0], [0, 160], [360, 0], [360, 160]]).astype(np.float32)
    src= np.array([lu, ru, ld, ld]).astype(np.float32)

    projective_matrix = cv2.getPerspectiveTransform(src, dst)
    croped = cv2.warpPerspective(img, projective_matrix, (360,160))
    return croped


def detect(dataPath, clf, t=10):
    """
    Please read detectData.txt to understand the format. 
    Use cv2.VideoCapture() to load the video.gif.
    Use crop() to crop each frame (frame size = 1280 x 800) of video to get parking space images. (image size = 360 x 160) 
    Convert each parking space image into 36 x 16 and grayscale.
    Use clf.classify() function to detect car, If the result is True, draw the green box on the image like the example provided on the spec. 
    Then, you have to show the first frame with the bounding boxes in your report.
    Save the predictions as .txt file (Adaboost_pred.txt), the format is the same as GroundTruth.txt. 
    (in order to draw the plot in Yolov5_sample_code.ipynb)
    
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    positions = [] 
    if not os.path.exists(dataPath):
        print("Error: detectData.txt not found.")
        return
    with open(dataPath) as file:
        num_of_parking = int(file.readline()) 
        #print(num_of_parking," num_of_parking")
        for _ in range(num_of_parking):
            temp = file.readline() 
            temp = temp.split(" ") 
            res = tuple(map(int, temp)) 
            positions.append(res) 
    video_path = "data/detect/video.gif"
    if not os.path.exists(video_path):
        print("Error: video.gif not found.")
        return
    cap = cv2.VideoCapture(video_path)
    diff=1
    frames = 1 
    ogif = [] 
    while True:
        diff_set = []
        
        _, img = cap.read() 
        if img is None:
            break
        for object in positions: 
            pic = crop(*object, img) 
            pic = cv2.resize(pic, (36, 16)) 
            pic = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY) 
            diff_set.append(clf.classify(pic))
            #if diff==1:
                #print(object, " ",diff_set[-1])
        #if diff == 1:
           #diff=0
        for i, label in enumerate(diff_set):
            if label:
                pos = [[positions[i][idx], positions[i][idx+1]] for idx in range(0, 8, 2)] 
                pos[2], pos[3] = pos[3], pos[2] 
                pos = np.array(pos, np.int32)
                cv2.polylines(img, [pos], color=(0, 255, 0), isClosed=True)
                    
        ogif.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        frames += 1 
    if ogif:
        imageio.mimsave(f'results.gif', ogif, fps=2)
        #print("done")
    else:
        print("No frames found to write to GIF file.")
    # End your code (Part 4)