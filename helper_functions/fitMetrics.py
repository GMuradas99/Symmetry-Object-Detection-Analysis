import cv2
import numpy as np

def intersection_over_union(mask: np.ndarray, points: list) -> float:
    """
    Calculate the Intersection over Union (IoU) between a binary mask and a polygon defined by points.
    Args:
        mask (np.ndarray): A binary mask where the object is represented by 1s and the background by 0s.
        points (list): A list of (x, y) tuples representing the vertices of the polygon.
    Returns:
        float: The IoU value, which is the ratio of the area of intersection to the area of union between the mask and the polygon.
    """

    # Create a binary mask from the polygon points
    poly_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(points, dtype=np.int32)], 1)

    # Calculate intersection and union
    intersection = np.logical_and(mask, poly_mask).sum()
    union = np.logical_or(mask, poly_mask).sum()

    # Compute IoU
    iou = intersection / union if union != 0 else 0
    return iou

def dice_score(mask: np.ndarray, points: list) -> float:
    """
    Calculate the Dice score between a binary mask and a polygon defined by a list of points.
    The Dice score is a measure of overlap between two binary masks, ranging from 0 (no overlap) 
    to 1 (perfect overlap).
    Parameters:
    mask (np.ndarray): A binary mask where the object is represented by 1s and the background by 0s.
    points (list): A list of (x, y) tuples representing the vertices of the polygon.
    Returns:
    float: The Dice score, a value between 0 and 1 indicating the degree of overlap.
    """

    # Create a binary mask from the polygon points
    poly_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(points, dtype=np.int32)], 1)

    # Calculate intersection and union
    intersection = np.logical_and(mask, poly_mask).sum()
    union = np.logical_or(mask, poly_mask).sum()

    # Compute Dice score
    dice = 2 * intersection / (mask.sum() + poly_mask.sum())
    return dice

def hausdorff_distance(mask: np.ndarray, points: list) -> float:
    """
    Calculate the Hausdorff distance between a binary mask and a polygon defined by a list of points.
    The Hausdorff distance is a measure of the maximum distance between two sets of points.
    Parameters:
    mask (np.ndarray): A binary mask where the object is represented by 1s and the background by 0s.
    points (list): A list of (x, y) tuples representing the vertices of the polygon.
    Returns:
    float: The Hausdorff distance between the mask and the polygon.
    """

    # Create a binary mask from the polygon points
    poly_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(poly_mask, [np.array(points, dtype=np.int32)], 1)

    # Compute the distance transform for the mask and polygon
    mask_dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    poly_dist = cv2.distanceTransform(poly_mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Compute the Hausdorff distance
    hausdorff_1 = np.max(mask_dist * poly_mask)
    hausdorff_2 = np.max(poly_dist * mask)
    hausdorff = max(hausdorff_1, hausdorff_2)
    return hausdorff