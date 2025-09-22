import os
import cv2
import numpy as np
def load_images(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples. 
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.) 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    set = []
    non_car_path=os.path.join(dataPath, "non-car")
    car_path=os.path.join(dataPath, "car")

    for i in os.listdir(non_car_path):
        #print(os.path.join(non_car_path, i))
        img = cv2.imread(os.path.join(non_car_path, i))
        img = cv2.resize(img, (36, 16))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data = (img, 0) 
        set.append(data)

    for i in os.listdir(car_path):
        img = cv2.imread(os.path.join(car_path, i))
        img = cv2.resize(img, (36, 16)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        data = (img, 1) 
        set.append(data) 
    
    
    # End your code (Part 1)
    
    return set