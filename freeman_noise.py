# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:40:03 2018
This file contains the implementation of the Freeman's chain code 
representation. The code is inspired by the following sources:
    - Digit Recognition Using Freeman Chain Code, by Pulipati Annapurna, 
    Sriraman Kothuri, Srikanth Lukka
    - Implementation by mcburger on Kaggle
    
The function assumes that the image has already been preprocessed
@author: eric
"""

import cv2 as cv
import pickle
import preprocess_image
from matplotlib import pyplot as plt


def freeman_chain_code(image_matrix, img_type='normal'):
    """Computes the Freeman code of an image.
    
    Input:
        image_matrix: numpy.ndarray, shape (image_height, image_width)
            contains the pixel values of an image
        format: string, ['normal', 'normalized']
            Tells whether the image we are using is a normal RGB one or a
            normalized one
        
    Outputs:
        - fr_chain: string
            Freeman chain code
        - boundaries: list of tuples
            List of points located on the contours of 
        the image, List"""
    
    # Change pixel values to binary (a convention to make the computation
    # easier). White pixel = 0, Black pixel = 255
    # Typically, the black pixel is 1 instead of 255 but I used it for
    # displaying convenience
    t = 25  # First 25
    if img_type == 'normalized':
        t = 0.5
    
    # Preprocessing the image
    # Denoising (to handle "stains")
    #image_matrix = cv.medianBlur(image_matrix, 7)
    
    # Thresholding the image
   # _, image_matrix = cv.threshold(image_matrix, t,
  #                                   255, cv.THRESH_BINARY)
    image_matrix = preprocess_image.image_smoother(image_matrix, t)
  
  
    # Detect the starting pixel
    start_point = None
   
    for i, row in enumerate(image_matrix):
        for j, pixel_value in enumerate(row):
            if pixel_value == 255:
                start_point = i, j
#                print("First pixel:", start_point, pixel_value)
                break
        if start_point is not None:
            # At this stage in the algorithm, we already found the starting
            # pixel
            break
    
   
    
    # If there is a start point, compute the Freeman code
        
    # Freeman's directions by testing order
    fr_directions = [0, 1, 2, 3, 4, 5, 6, 7]
    change = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0),
             (1, -1), (0, -1), (-1, -1)]
    #print("Directions initialized")
    
    # Freeman's chain code
    fr_chain = []
    
    # Coordinates of the pixels in the boundaries. Can be used to plot the
    # boundaries
    boundaries = []
    boundaries.append(start_point)
    
    curr_point = start_point
    count = 0
    
    #print("Locating the contours...")
    while True:
        count += 1
        #print(fr_directions)
        for direction in fr_directions:
            coord_shift = change[direction]
            new_point = (curr_point[0] + coord_shift[0], 
                         curr_point[1] + coord_shift[1])
            if new_point[0] < image_matrix.shape[0] and \
            new_point[1] < image_matrix.shape[1] and \
            image_matrix[new_point] == 255:  # To avoid infinite loops
                #print(new_point)
                curr_point = new_point
                boundaries.append(new_point)
                fr_chain.append(direction)
                break
            
        if curr_point == start_point:
            break
            
        start_direction = (fr_chain[-1] + 5) % 8
        fr_directions1 = range(start_direction, 8)
        fr_directions2 = range(0, start_direction)
        fr_directions = []
        fr_directions.extend(fr_directions1)
        fr_directions.extend(fr_directions2)
        
    
  
        
    return [str(i) for i in fr_chain], boundaries


def get_image_from_file(path_to_image):
    """Get an image from file at location path_to_image. Proceeds with
    inverting black and white pixels values (now 0 for white, 255 for black).
    This operation is performed for convenience"""
    
    # The parameter 0 is so as to read the image as a grayscale image
    image_matrix = cv.imread(path_to_image, 0)
    # Switch black and white pixels values
    image_matrix = cv.bitwise_not(image_matrix)
    
    return image_matrix

def freeman_from_image(path_to_image):
    """Computes the Freeman chain code of the image located at path_to_image"""
    
    image_matrix = get_image_from_file(path_to_image)
    return freeman_chain_code(image_matrix)
    #return freeman_chain_code(image_matrix)


def freeman_to_points(start_point, freeman_code):
    """Retrieves the boundary points from an image freeman code and the start
    point"""
    
    change = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0),
             (1, -1), (0, -1), (-1, -1)]
    
    boundaries = []
    current_point = start_point
    
    for i in freeman_code:
        new_point = (current_point[0] + change[i, 0], 
                     current_point[1] + change[i, 1])
        current_point = new_point
        boundaries.append(new_point)
        
    return boundaries


def save_to_file(freeman_codes, path_to_file):
    """Save the freeman codes in the input list to the specified path
    
    Parameters
    ----------
    freeman_codes: list
        List containing freeman codes
        
    path_to_file: string
        Path to the file where the list will be saved
    """
    
    with open(path_to_file, "wb") as out:
        pickle.dump(freeman_codes, out)
        

def freemans_from_file(path_to_file):
    """Recovers freeman codes saved to the file at the specified location and
    returns a list containing them
    """
    
    with open(path_to_file, "rb") as input_file:
        freeman_codes = pickle.load(input_file)
        
    return freeman_codes


def plot_boundaries(image_matrix, boundaries, color='yellow'):
    """Plot an image (a digit image most likely) and its countour"""
    
    # Plot the image
    plt.imshow(image_matrix, cmap="Greys")
    # Plot the countours
    plt.plot([pixel[1] for pixel in boundaries],
         [pixel[0] for pixel in boundaries],
         color=color)
    plt.show()