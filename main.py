import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import operator
from math import floor,ceil
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import os
from grab import *
from extract import *
from solver import *

def solve_complete(PATH):
    img = cv.imread(PATH, cv.IMREAD_GRAYSCALE)
    processed = proc_img(img)
    corners = grid(processed)
    img_final = perspect(img,corners)
    imgp = extract_grid(img_final)
    sudoku = extract_digit(imgp)
    print(sudoku)
    solved = sudoku_solver(sudoku)
    print(solved)
    return solved
