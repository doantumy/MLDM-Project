# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 18:30:46 2018
This file contains functions to preprocess an image 
@author: deric
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def get_sep_data(dataset):
    """Set the column 'label' and the pixels apart.
    Returns:
        -labels
        -pixels
    """
    
    labels = dataset['label']
    dataset_images = dataset.drop('label', axis=1)
    
    return labels, dataset_images


def vec_to_matrix(row):
    """Transform a vector representation of an image into a matrix one"""
    
    image = np.reshape(row, (-1, 28)).astype(np.uint8)
    
    return image


def image_smoother(img, t=70):
    """
    Parameters
        img: numpy.array
            Matrix containing the pixel values of an image (eventually
            containing noise)
            
    Returns
        denoised: numpy.array
            img without noise
    """
    
     # Denoising (to handle "stains")
    img = cv.medianBlur(img, 5)
    # Thresholding the image
    _, img = cv.threshold(img, t, 255, cv.THRESH_BINARY)
    
    return img


def plot_preprocessed(img):
    """Apply the preprocessing and plot the result"""
    
    img = image_smoother(img, 25)
    plt.imshow(img, cmap="Greys")
    plt.show()