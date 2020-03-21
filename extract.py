import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import operator
from math import floor,ceil
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os

def extract_grid(img):
    row, col = img.shape
    rowp = row/9
    colp = col/9
    imgp = []
    dim = (floor(rowp),floor(colp))
    for j in range(9):
        for i in range(9):
            pts1 = np.float32([[i*1.025*rowp,j*1.025*colp],[(i+1)*0.975*rowp,j*1.025*colp],[i*1.025*rowp,(j+1)*0.975*colp],[(i+1)*0.975*rowp,(j+1)*0.975*colp]])
            pts2 = np.float32([[0,0],[rowp,0],[0,colp],[rowp,colp]])
            M = cv.getPerspectiveTransform(pts1,pts2)
            dst = cv.warpPerspective(img,M,dim)
            imgp.append(dst)
    return imgp

def identify_digit(box):
    model = tf.keras.models.load_model('./models/model.h5')
    temp1 = cv.medianBlur(box.copy(), 3)
    temp1 = cv.adaptiveThreshold(temp1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    temp1 = cv.bitwise_not(temp1, temp1)
    temp1 = cv.resize(temp1, (50,50), interpolation = cv.INTER_AREA)
    temp1 = temp1.reshape(1,50,50, 1)
    temp1 = temp1.astype('float32')
    temp1 /= 255
    predict = model.predict(temp1)
    return(np.argmax(predict))

def extract_digit(imgp):
    sudoku = []
    temp = []
    for i in range(0,81):
        num = identify_digit(imgp[i])
        temp.append(num)
        if((i+1)%9 == 0):
            sudoku.append(temp.copy())
            temp = []
