import cv2
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops

import helper_functions.displayFunctions as display

def glcm(img_color: np.ndarray, mask: np.ndarray) -> float:
    """
    Computes the Gray Level Co-occurrence Matrix (GLCM) and calculates the homogeneity of the given image within the specified mask.
    Parameters:
    img_color (numpy.ndarray): The input color image in RGB format.
    mask (numpy.ndarray): The binary mask to apply to the image. Should be the same size as img_color.
    Returns:
    float: The mean homogeneity value computed from the GLCM.
    """
    # Greyscale image
    img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # Apply the mask to the grayscale image and normalize
    masked_image = img * mask
    masked_image_uint8 = (masked_image * 255).astype(np.uint8)
    masked_pixels = masked_image_uint8[mask == 1]

    # Compute GLCM on masked pixels (reshape for GLCM input)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 0]
    glcm = graycomatrix(masked_pixels.reshape(-1, 1), distances, angles, levels=256, symmetric=True, normed=True)

    # Calculate homogeneity
    homogeneity = graycoprops(glcm, 'homogeneity')
    return homogeneity.mean()

def otsus_method(img_orig: np.ndarray, mask: np.ndarray, row: pd.core.series.Series) -> float:
    """
    Applies Otsu's thresholding method to an image and calculates the percentage of background within a masked region.
    Parameters:
    img_orig (np.ndarray): The original image in BGR format.
    mask (np.ndarray): A binary mask where the region of interest is marked.
    row (pd.core.series.Series): A pandas Series containing the coordinates for cropping the image.
    Returns:
    float: The percentage of background within the masked region.
    """

    gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_cut = display.crop_image(threshold, row)

    # Apply the mask to the thresholded image
    threshold_cut_mask = threshold_cut * mask

    percentage_of_background = np.sum(threshold_cut_mask) / np.sum(mask)

    return percentage_of_background

def rms(img_color: np.ndarray, mask: np.ndarray) -> float:
    """
    Calculate the root mean square (RMS) contrast of a given image within a specified mask.
    Parameters:
    img_color (np.ndarray): The input color image in BGR format.
    mask (np.ndarray): A binary mask where the RMS contrast is to be calculated. 
                       The mask should have the same width and height as the input image.
    Returns:
    float: The RMS contrast value of the image within the masked region.
    """

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    mean_intensity = np.mean(gray[mask == 1])
    rms_contrast = np.sqrt(np.mean((gray[mask == 1] - mean_intensity) ** 2))

    return rms_contrast

def edge_density(img_orig: np.ndarray, mask: np.ndarray, row: pd.core.series.Series, threshold1: int = 100, threshold2: int = 200) -> float:
    """
    Calculate the edge density of an image within a specified region of interest.
    Parameters:
    img_orig (np.ndarray): The original image in BGR format.
    mask (np.ndarray): A binary mask where the region of interest is marked with 1s and the rest with 0s.
    row (pd.core.series.Series): A pandas Series containing the coordinates for cropping the image.
    threshold1 (int, optional): The first threshold for the hysteresis procedure in Canny edge detection. Default is 100.
    threshold2 (int, optional): The second threshold for the hysteresis procedure in Canny edge detection. Default is 200.
    Returns:
    float: The edge density, calculated as the ratio of edge pixels to the total number of pixels in the masked region.
    """
    
    # Convert image to grayscale
    gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

    # Compute the Canny edge detection on the grayscale image
    edges_orig = cv2.Canny(gray_orig, threshold1, threshold2)
    edges_orig_cut = display.crop_image(edges_orig, row)
    edges = np.where(edges_orig_cut > 0, 1, 0)

    # Apply the mask to the edges
    edges = edges * mask

    # Calculate the edge density
    edge_density = np.sum(edges) / np.sum(mask)
    return edge_density