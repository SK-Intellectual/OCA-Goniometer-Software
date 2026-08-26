# Import necessary libraries

import io
import cv2
import math
import os
import numpy as np
import csv
import scipy
import warnings
from pathlib import Path
import matplotlib.animation as animation
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import defaultdict
from ADB_extractor import video_retrieval_from_phone
from Thresholds import get_threshes
from scipy.optimize import least_squares
# warnings.filterwarnings("ignore", category=np.RankWarning)

def filter_points_for_fitting(all_points, intersection_point_x, side, num_points=40):
    """
    Filters and selects a subset of points for fitting, based on proximity to the
    intersection point. This avoids fitting on inner points or noise.

    Args:
        all_points (np.array): All contour points.
        intersection_point_x (float): The x-coordinate of the intersection.
        side (str): 'left' or 'right' side of the droplet.
        num_points (int): Number of points to select for the fit.

    Returns:
        np.array: The filtered subset of points.
    """
    # Sort points by horizontal distance from the intersection point
    distances = np.abs(all_points[:, 0] - intersection_point_x)
    sorted_indices = np.argsort(distances)
    
    # Select the `num_points` closest to the intersection
    filtered_points = all_points[sorted_indices[:num_points]]
    
    # Ensure points are sorted by x-coordinate for the fit function
    return filtered_points[np.argsort(filtered_points[:, 0])]

def validate_circle_fit(points, x0, y0, r, image_width=None, image_height=None):
    """
    Validates a fitted circle to detect degenerate or ill-conditioned fits.
    
    Rejects fits that are:
    - Unphysically large or small radius
    - Poor fit quality (high residuals)
    - Near-collinear points
    - Intersections outside image bounds
    
    Args:
        points (np.array): Points used for circle fitting
        x0, y0, r: Fitted circle parameters
        image_width, image_height: Image dimensions for bounds checking
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Estimate droplet width from points
    if len(points) > 0:
        droplet_width = np.max(points[:, 0]) - np.min(points[:, 0])
    else:
        droplet_width = 100  # Default fallback
    
    # Check 1: Radius bounds - DISABLED (droplet_width calculated incorrectly)
    # r_min = 10  # Minimum 10 pixels
    # r_max = 5 * droplet_width  # Maximum 5x droplet width
    # 
    # if r < r_min:
    #     return False, f"Radius too small: {r:.1f} < {r_min}"
    # if r > r_max:
    #     return False, f"Radius too large: {r:.1f} > {r_max:.1f}"
    
    # Check 2: Fit residuals (RMS error)
    distances = np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2)
    residuals = np.abs(distances - r)
    rms_error = np.sqrt(np.mean(residuals**2))
    
    # Threshold: residual should be < 5 pixels for good fit
    if rms_error > 5.0:
        return False, f"Poor fit quality: RMS error = {rms_error:.2f} px"
    
    # Check 3: Collinearity - DISABLED (too strict for valid droplet curves)
    # try:
    #     # Simplified collinearity check: variance in distances from center
    #     distance_std = np.std(distances)
    #     if distance_std < 1.0:  # Points are too uniformly distributed = suspicious
    #         return False, "Points appear collinear (low distance variance)"
    # except:
    #     pass  # Skip if check fails
    
    # Check 4: Intersection points within reasonable bounds
    # For a droplet, intersection points should be near the fitted points
    if image_width is not None:
        # Check if circle center is absurdly far from image
        if abs(x0) > 10 * image_width or abs(y0) > 10 * image_width:
            return False, f"Circle center too far from image: ({x0:.1f}, {y0:.1f})"
    
    # All checks passed
    return True, "Valid fit"

def fit_line_and_calculate_angle(points, baseline_y, side, prev_contact_point=None, max_displacement=float('inf')):
    """
    Fallback function for collinear points: fit a line instead of circle.
    Used when points are nearly collinear and circle fitting produces degenerate results.
    
    Uses PCA-based Total Least Squares to handle vertical/near-vertical lines.
    
    Args:
        points: Array of (x, y) coordinates
        baseline_y: Y-coordinate of baseline
        side: 'left' or 'right'
        prev_contact_point: Previous contact point for temporal continuity
        max_displacement: Maximum allowed displacement
    
    Returns:
        dict: Same format as calculate_contact_angle_from_circle
    """
    if len(points) < 2:
        return {'valid': False, 'error': f"[{side.upper()}] Not enough points for line fit"}
    
    # Limit to 15 points closest to baseline for more stable fitting
    max_line_fit_points = 15
    if len(points) > max_line_fit_points:
        # Sort by y-coordinate in descending order (closest to baseline first, assuming y increases downward)
        sorted_indices = np.argsort(points[:, 1])[::-1]
        points = points[sorted_indices[:max_line_fit_points]]
    
    # PCA-based line fit (Total Least Squares)
    # This works even when x values are identical (vertical line)
    mean = np.mean(points, axis=0)
    centered = points - mean
    
    # Compute covariance matrix
    cov = np.cov(centered.T)
    
    # Find principal component (eigenvector with largest eigenvalue)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argmax(eigenvalues)
    direction = eigenvectors[:, idx]  # [dx, dy]
    
    # Ensure direction points in positive x direction for consistency
    if direction[0] < 0:
        direction = -direction
    
    # Line equation: point = mean + t * direction
    # Find intersection with baseline: mean[1] + t * direction[1] = baseline_y
    # Solve for t: t = (baseline_y - mean[1]) / direction[1]
    
    if abs(direction[1]) < 1e-10:
        # Line is horizontal, no intersection with baseline
        return {'valid': False, 'error': f"[{side.upper()}] Line is horizontal, no baseline intersection"}
    
    t = (baseline_y - mean[1]) / direction[1]
    intersection_x = mean[0] + t * direction[0]
    intersection_point = np.array([intersection_x, baseline_y])
    
    # Check temporal continuity if we have previous point
    if prev_contact_point is not None:
        displacement = np.linalg.norm(intersection_point - np.array(prev_contact_point))
        if displacement > max_displacement:
            return {
                'valid': False,
                'error': f"[{side.upper()}] Line fit displacement {displacement:.2f}px > {max_displacement}px"
            }
    
    # Calculate contact angle from line direction
    # slope = dy/dx = direction[1] / direction[0]
    if abs(direction[0]) < 1e-10:
        # Vertical line → 90 degrees
        contact_angle = 90.0
    else:
        slope = direction[1] / direction[0]
        angle_rad = math.atan(slope)
        angle_deg = math.degrees(angle_rad)
        
        # Determine if obtuse based on slope direction
        if side == 'left':
            # Left side: negative slope → acute, positive slope → obtuse
            if slope < 0:
                contact_angle = abs(angle_deg)
            else:
                contact_angle = 180 - abs(angle_deg)
        else:  # right side
            # Right side: positive slope → acute, negative slope → obtuse
            if slope > 0:
                contact_angle = abs(angle_deg)
            else:
                contact_angle = 180 - abs(angle_deg)
        
        # Ensure angle in valid range
        contact_angle = max(0, min(180, contact_angle))
    
    print(f"[{side.upper()}] Using LINE FIT (PCA): angle={contact_angle:.2f}°")
    
    return {
        'valid': True,
        'angle': contact_angle,
        'contact_point': tuple(intersection_point),
        'circle_params': None,  # No circle for line fit
        'displacement': np.linalg.norm(intersection_point - np.array(prev_contact_point)) if prev_contact_point else 0.0,
        'error': None,
        'fit_type': 'line',
        'tangent_slope': slope if abs(direction[0]) >= 1e-10 else float('inf')  # For visualization
    }



def fit_circle_to_points(points, baseline_y=None):
    """
    Fits a circle to a set of 2D points using weighted least squares optimization.
    Points closer to the baseline are weighted more heavily.
    
    Args:
        points (np.array): Array of shape (n, 2) containing [x, y] coordinates.
        baseline_y (float): Y-coordinate of baseline for weighted fitting.
                           If None, uses unweighted fitting.
    
    Returns:
        tuple: (x0, y0, r) where (x0, y0) is the circle center and r is the radius.
               Returns None if fitting fails.
    """
    if len(points) < 3:
        print("Need at least 3 points to fit a circle.")
        return None
    
    # Calculate weights based on distance from baseline
    weights = None
    if baseline_y is not None:
        # Distance from baseline (vertical distance)
        distance_from_baseline = np.abs(points[:, 1] - baseline_y)
        
        # Adaptive decay factor: proportional to range of y-coordinates
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        decay_factor = max(y_range / 3.0, 10.0)  # At least 10 pixels
        
        # Exponential decay: points near baseline get weight ~1, far points get weight ~0
        weights = np.exp(-distance_from_baseline / decay_factor)
        
        # Normalize weights to maintain scale (optional but helps numerics)
        weights = weights / np.mean(weights)
    
    # Initial guess for circle center (mean of points) and radius
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    
    # Initial radius estimate
    distances = np.sqrt((points[:, 0] - x_mean)**2 + (points[:, 1] - y_mean)**2)
    r_initial = np.mean(distances)
    
    # Define the residual function
    def residuals(params):
        x0, y0, r = params
        # Distance from each point to the circle
        distances = np.sqrt((points[:, 0] - x0)**2 + (points[:, 1] - y0)**2)
        residual_values = distances - r
        
        # Apply weights if provided (weighted residuals)
        if weights is not None:
            residual_values = residual_values * np.sqrt(weights)
        
        return residual_values
    
    # Initial parameters
    initial_params = [x_mean, y_mean, r_initial]
    
    # Perform least squares optimization
    result = least_squares(residuals, initial_params)
    
    if result.success:
        x0, y0, r = result.x
        return x0, y0, abs(r)  # Ensure radius is positive
    else:
        print("Circle fitting failed.")
        return None


def calculate_contact_angle_from_circle(baseline_y, points, side='left', prev_contact_point=None, 
                                         image_width=None, max_displacement=3.0):
    """
    Calculates the contact angle using TEMPORAL CONTINUITY to select the intersection point.
    
    CRITICAL CHANGE: Does NOT use binary center-based switching. Instead, selects the 
    intersection point closest to the previous frame's contact point.
    
    Args:
        baseline_y (float): Y-coordinate of the baseline.
        points (np.array): Array of shape (n, 2) containing [x, y] coordinates of the droplet edge.
        side (str): 'left' or 'right' - which side of the droplet.
        prev_contact_point (tuple): (x, y) coordinates of contact point from previous frame
        image_width (float): Image width for validation checks
        max_displacement (float): Maximum allowed displacement between frames (pixels)
    
    Returns:
        dict: {
            'angle': contact angle in degrees,
            'contact_point': (x, y) selected contact point,
            'circle_params': (x0, y0, r),
            'valid': True if fit is valid, False if should be rejected
            'error': error message if rejected
        }
   """
    # Fit circle to points with weighted fitting (prioritize points near baseline)
    fit_result = fit_circle_to_points(points, baseline_y=baseline_y)
    
    if fit_result is None:
        return {'valid': False, 'error': f"Failed to fit circle for {side} side"}
    
    x0, y0, r = fit_result
    
    # COLLINEARITY DETECTION: If radius is too large, points are nearly collinear
    # Fall back to line fitting instead of using degenerate circle
    # Threshold: radius > 5000 pixels indicates degenerate fit
    COLLINEARITY_THRESHOLD = 5000  # pixels

    
    if r > COLLINEARITY_THRESHOLD:
        print(f"[{side.upper()}] Circle radius {r:.1f}px > {COLLINEARITY_THRESHOLD}px, falling back to LINE FIT")
        return fit_line_and_calculate_angle(
            points, baseline_y, side,
            prev_contact_point=prev_contact_point,
            max_displacement=max_displacement
        )
    
    # VALIDATION: Check if circle fit is degenerate
    is_valid, validation_msg = validate_circle_fit(points, x0, y0, r, image_width=image_width)
    if not is_valid:
        # If validation fails, try line fit as fallback
        print(f"[{side.upper()}] Circle validation failed ({validation_msg}), trying LINE FIT")
        return fit_line_and_calculate_angle(
            points, baseline_y, side,
            prev_contact_point=prev_contact_point,
            max_displacement=max_displacement
        )
    
    #  ============================================================================
    # COMPUTE BOTH INTERSECTION POINTS (NO ASSUMPTIONS ABOUT WHICH IS CORRECT)
    # ============================================================================
    dy = baseline_y - y0
    
    # Check if circle intersects baseline
    if abs(dy) > r:
        return {'valid': False, 'error': f"[{side.upper()}] Circle does not intersect baseline"}
    
    # Calculate both intersection points
    dx = np.sqrt(r**2 - dy**2)
    intersection_left = np.array([x0 - dx, baseline_y])
    intersection_right = np.array([x0 + dx, baseline_y])
    
    # ============================================================================
    # TEMPORAL CONTINUITY: SELECT INTERSECTION CLOSEST TO PREVIOUS FRAME
    # ============================================================================
    if prev_contact_point is not None:
        prev_point = np.array(prev_contact_point)
        
        # Calculate distances to both candidates
        dist_to_left = np.linalg.norm(intersection_left - prev_point)
        dist_to_right = np.linalg.norm(intersection_right - prev_point)
        
        # Select the closest one
        if dist_to_left < dist_to_right:
            intersection_point = intersection_left
            min_displacement = dist_to_left
        else:
            intersection_point = intersection_right
            min_displacement = dist_to_right
        
        # FRAME REJECTION: Check if displacement is too large
        if min_displacement > max_displacement:
            return {
                'valid': False,
                'error': f"[{side.upper()}] Displacement {min_displacement:.2f}px > {max_displacement}px threshold"
            }
    else:
        # FIRST FRAME: Use improved outer fit detection based on fitted points
        # Find the point with maximum y-coordinate (closest to baseline = contact point)
        max_y_idx = np.argmax(points[:, 1])
        contact_point_x = points[max_y_idx, 0]
        
        # Determine if outer fit based on circle center position relative to contact point
        if side == 'right':
            # For right side: if center is to the RIGHT of the contact point → outer fit
            if x0 > contact_point_x:
                # Outer fit: choose LEFT intersection (lower x)
                intersection_point = intersection_left
                print(f"[{side.upper()}] First frame outer fit: center x={x0:.1f} > contact x={contact_point_x:.1f}, using LEFT intersection")
            else:
                # Inner fit: choose RIGHT intersection (higher x)
                intersection_point = intersection_right
                print(f"[{side.upper()}] First frame inner fit: using RIGHT intersection")
        else:  # left side
            # For left side: if center is to the LEFT of the contact point → outer fit
            if x0 < contact_point_x:
                # Outer fit: choose RIGHT intersection (higher x)
                intersection_point = intersection_right
                print(f"[{side.upper()}] First frame outer fit: center x={x0:.1f} < contact x={contact_point_x:.1f}, using RIGHT intersection")
            else:
                # Inner fit: choose LEFT intersection (lower x)
                intersection_point = intersection_left
                print(f"[{side.upper()}] First frame inner fit: using LEFT intersection")
        
        min_displacement = 0.0

    
    # ============================================================================
    # CALCULATE CONTACT ANGLE FROM SELECTED INTERSECTION
    # ============================================================================
    # Vector from circle center to intersection point (radius vector)
    dx_vector = intersection_point[0] - x0
    dy_vector = intersection_point[1] - y0
    
    # The tangent to the circle at the intersection point is perpendicular to the radius
    if side == 'left':
        tangent_x = -dy_vector
        tangent_y = dx_vector
    else:  # right side
        tangent_x = dy_vector
        tangent_y = -dx_vector
    
    # Calculate the angle of the tangent vector with respect to horizontal (baseline)
    tangent_angle_rad = math.atan2(tangent_y, tangent_x)
    
    # The slope from the angle
    if abs(math.cos(tangent_angle_rad)) > 1e-10:
        tangent_slope = math.tan(tangent_angle_rad)
    else:
        tangent_slope = float('inf')  # Vertical tangent
    
    #  ============================================================================
    # Acute/Obtuse Detection Using Circle Center Position
    # ============================================================================
    center_above_baseline = (y0 < baseline_y)  # y increases downward in images
    is_obtuse = center_above_baseline
    
    # Detect outer fit by checking if we selected the "opposite" intersection
    # For left side: if we selected RIGHT intersection → outer fit
    # For right side: if we selected LEFT intersection → outer fit
    is_outer_fit = False
    if side == 'left' and np.allclose(intersection_point, intersection_right):
        is_outer_fit = True
    elif side == 'right' and np.allclose(intersection_point, intersection_left):
        is_outer_fit = True
    
    # Override obtuse detection for outer fits - they are ALWAYS acute
    if is_outer_fit:
        is_obtuse = False
    
    # Calculate the contact angle
    angle_from_horizontal = math.degrees(math.atan(tangent_slope))
    
    # Adjust based on side and obtuseness
    if side == 'left':
        if is_obtuse:
            contact_angle = 180 + angle_from_horizontal if angle_from_horizontal < 0 else 180 - angle_from_horizontal
        else:
            contact_angle = -angle_from_horizontal if angle_from_horizontal < 0 else angle_from_horizontal
            
    else:  # right side
        if is_obtuse:
            contact_angle = 180 - abs(angle_from_horizontal)
        else:
            contact_angle = abs(angle_from_horizontal)
    
    # For outer fits, if calculated angle is obtuse, convert to acute
    if is_outer_fit and contact_angle > 90:
        contact_angle = 180 - contact_angle
        print(f"[{side.upper()}] Outer fit angle adjustment: forcing acute angle = {contact_angle:.2f}°")
    
    # Ensure angle is in valid range [0, 180]
    contact_angle = max(0, min(180, contact_angle))

    
    # Return comprehensive result dictionary
    return {
        'valid': True,
        'angle': contact_angle,
        'contact_point': tuple(intersection_point),
        'circle_params': (x0, y0, r),
        'displacement': min_displacement,
        'error': None
    }


def calculate_contact_angles_circle_fit(baseline_y, left_points, right_points, 
                                         prev_left_contact=None, prev_right_contact=None,
                                         image_width=None, max_displacement=3.0):
    """
    Main function to calculate both left and right contact angles using circle fitting
    with temporal continuity.
    
    Args:
        baseline_y (float): Y-coordinate of the baseline.
        left_points (np.array): Points on the left side of the droplet, shape (n, 2).
        right_points (np.array): Points on the right side of the droplet, shape (n, 2).
        prev_left_contact (tuple): Previous left contact point (x, y) for temporal continuity
        prev_right_contact (tuple): Previous right contact point (x, y) for temporal continuity
        image_width (float): Image width for validation
        max_displacement (float): Maximum allowed frame-to-frame displacement
    
    Returns:
        dict: {
            'left_result': result dictionary from left fit,
            'right_result': result dictionary from right fit,
            'both_valid': True if both sides are valid
        }
    """
    # Calculate left side
    left_result = calculate_contact_angle_from_circle(
        baseline_y, left_points, side='left', 
        prev_contact_point=prev_left_contact,
        image_width=image_width,
        max_displacement=max_displacement
    )
    
    # Calculate right side
    right_result = calculate_contact_angle_from_circle(
        baseline_y, right_points, side='right',
        prev_contact_point=prev_right_contact,
        image_width=image_width,
        max_displacement=max_displacement
    )
    
    return {
        'left_result': left_result,
        'right_result': right_result,
        'both_valid': left_result['valid'] and right_result['valid']
    }


def preprocess_contour_points(points, baseline_y):
    """
    Preprocesses contour points by removing internal points using an ellipse filter.
    
    Creates an ellipse inside the droplet and removes points within it, keeping only
    boundary points.
    
    Args:
        points (np.array): Array of shape (n, 2) containing [x, y] coordinates.
        baseline_y (float): Y-coordinate of the baseline (contact line).
    
    Returns:
        tuple: (filtered_points, ellipse_params)
    """
    if len(points) == 0:
        return points, None
    
    # Find the three reference points
    leftmost_idx = np.argmin(points[:, 0])
    rightmost_idx = np.argmax(points[:, 0])
    topmost_idx = np.argmin(points[:, 1])  # Assuming y increases downward
    
    leftmost_point = points[leftmost_idx]
    rightmost_point = points[rightmost_idx]
    topmost_point = points[topmost_idx]
    
    # Calculate the width (distance between leftmost and rightmost)
    width = rightmost_point[0] - leftmost_point[0]
    
    # Calculate ellipse endpoints
    left_ellipse_x = leftmost_point[0] + width / 10
    left_ellipse_y = leftmost_point[1]
    
    right_ellipse_x = rightmost_point[0] - width / 10
    right_ellipse_y = rightmost_point[1]
    
    # Ellipse center (horizontal midpoint, vertical at baseline)
    ellipse_center_x = (left_ellipse_x + right_ellipse_x) / 2
    ellipse_center_y = baseline_y  # Center on the baseline (contact line)
    
    # Semi-major axis (horizontal): half the distance between ellipse endpoints
    semi_major_a = (right_ellipse_x - left_ellipse_x) / 2
    
    # Calculate vertical distance for ellipse top
    height_from_top = topmost_point[1] - ellipse_center_y
    top_ellipse_y = topmost_point[1] + abs(height_from_top) / 8
    
    # Semi-minor axis (vertical): distance from center to top
    semi_minor_b = abs(ellipse_center_y - top_ellipse_y)
    
    # Filter points: keep only those OUTSIDE the ellipse
    normalized_x = (points[:, 0] - ellipse_center_x) / semi_major_a
    normalized_y = (points[:, 1] - ellipse_center_y) / semi_minor_b
    
    # Points outside the ellipse
    outside_ellipse = (normalized_x**2 + normalized_y**2) > 1.0
    
    filtered_points = points[outside_ellipse]
    
    # Store ellipse parameters for visualization
    ellipse_params = {
        'center': (ellipse_center_x, ellipse_center_y),
        'semi_major': semi_major_a,
        'semi_minor': semi_minor_b
    }
    
    return filtered_points, ellipse_params


# Global variables for cropping
ref_point = []
cropping = False
vertical_lines = [None, None]

# Mouse callback function to select the cropping area
def crop_image(event, x, y, flags, param):
    '''
    Handles mouse events for cropping an image. Tracks the region selected by the user
    using left mouse button events and dynamically updates the cropping area.

    Args:
        event: OpenCV mouse event (e.g., button press, movement, release).
        x, y: Coordinates of the mouse pointer during the event.
        flags: OpenCV flags for mouse event.
        param: Clone of the image to allow real-time rectangle drawing.
    '''
    global ref_point, cropping

    clone = param.copy()  # Copy the original image to draw the rectangle dynamically

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            # Draw a rectangle in real-time as the user selects the ROI
            current_point = (x, y)
            top_left = (min(ref_point[0][0], current_point[0]), min(ref_point[0][1], current_point[1]))
            bottom_right = (max(ref_point[0][0], current_point[0]), max(ref_point[0][1], current_point[1]))
            cv2.rectangle(clone, top_left, bottom_right, (0, 0, 255), 2)  # Draw the rectangle in red
            # Move the window (x, y position on screen)
            cv2.moveWindow("Crop Image", 0 , 0)
            cv2.imshow("Crop Image", clone)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        if len(ref_point) != 2:
            raise ValueError("Invalid cropping points. Please select a valid ROI.")
        
        # Draw the final rectangle around the region of interest
        top_left = (min(ref_point[0][0], x), min(ref_point[0][1], y))
        bottom_right = (max(ref_point[0][0], x), max(ref_point[0][1], y))
        cv2.rectangle(param, top_left, bottom_right, (0, 0, 255), 2)  # Draw the rectangle in red
        # Resize the window (width, height)
        # Move the window (x, y position on screen)
        cv2.moveWindow("Crop Image", 0 , 0)
        cv2.imshow("Crop Image", param)

# Function to resize the image to fit within the screen
def resize_to_fit_screen(image, screen_width=1920, screen_height=1080):
    """
    Resizes the image to fit within the screen dimensions while maintaining the aspect ratio.

    Args:
        image: The input image to resize.
        screen_width: Maximum width of the screen.
        screen_height: Maximum height of the screen.

    Returns:
        Resized image and the scale factor.
    """
    h, w = image.shape[:2]
    scale = min(screen_width / w, screen_height / h)
    if scale < 1:  # Resize only if the image is larger than the screen
        resized_image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return resized_image, scale
    return image, 1.0  # Return the original image if no resizing is needed

# Function to allow the user to crop the image
def get_cropped_image(image):
    '''
    Allows the user to interactively crop the input image using OpenCV.

    Args:
        image: Input image to be cropped.

    Returns:
        Cropped portion of the image or the original image if cropping is skipped.
    '''
    global ref_point,cropping
    
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    
    # Resize the image to fit the screen
    resized_image, scale = resize_to_fit_screen(image)

    clone = resized_image.copy()

    # Set up the window and set the mouse callback function for cropping
    cv2.namedWindow("Crop Image")
    # Resize the window (width, height)
    # Move the window (x, y position on screen)
    cv2.moveWindow("Crop Image", 0 , 0)
    cv2.setMouseCallback("Crop Image", crop_image, param=clone)

    # Display the image and wait for cropping
    while True:
        cv2.imshow("Crop Image", clone)
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset cropping
        if key == ord("r"):
            ref_point = []  # Clear previous ROI points
            # clone = image.copy()  # Reset the clone to the original image
            print("Reset cropping selection.")

        # Press 'c' to confirm the crop and break the loop
        elif key == ord("c") and len(ref_point) == 2:
            print("Cropping confirmed.")
            cropping=True
            break

        # Press 'n' to skip cropping
        elif key == ord("n"):
            cv2.destroyAllWindows()  # Close the window
            return image,scale  # Return the original image without cropping

        # Check if the window is closed
        if cv2.getWindowProperty("Crop Image", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            cv2.destroyAllWindows()
            return image,scale  # Return the original image without cropping

    # Close the cropping window after confirmation
    cv2.destroyAllWindows()

    # Crop the selected area
    if len(ref_point) == 2:
        x0, y0 = int(ref_point[0][0] / scale), int(ref_point[0][1] / scale)
        x1, y1 = int(ref_point[1][0] / scale), int(ref_point[1][1] / scale)
        # Get the coordinates in correct order
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        
        if x0 == x1 or y0 == y1:
            raise ValueError("Cropping region must have a non-zero area.")
        
        cropped_image = image[y0:y1, x0:x1]
        return cropped_image,scale
    else:
        return image,scale

def read_last_thresholds(file_path='thresholds_log.txt'):
    """Reads the last logged thresholds from the file."""
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # If the file is not empty, get the last line
            if lines:
                last_line = lines[-1].strip()
                # Extract the values of lower and upper thresholds
                lower_threshold = int(last_line.split('Final Lower Threshold: ')[1].split(',')[0])
                upper_threshold = int(last_line.split('Final Upper Threshold: ')[1].split(',')[0])
                return lower_threshold, upper_threshold
            else:
                print("No thresholds found in the file.")
                return None, None
    except FileNotFoundError:
        print("Thresholds log file not found.")
        return None, None

# Baseline selection from the first frame
def select_baseline(image):
    """
    Select the baseline y-position from the image by using a slider in a matplotlib plot.

    Args:
        image (numpy.ndarray): The input image for baseline selection.

    Returns:
        int: Selected baseline y-position.
    """
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Adjust the baseline using the slider')

    # Initial baseline y position
    init_baseline_y = image.shape[0] // 2

    # Create a slider for baseline adjustment
    ax_baseline = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    baseline_slider = Slider(ax_baseline, 'Baseline Y', 0, image.shape[0], valinit=init_baseline_y)

    # Draw the baseline
    line = ax.axhline(y=init_baseline_y, color='r', linestyle='--')

    # Update function for the slider
    def update(val):
        baseline_y = baseline_slider.val
        line.set_ydata([baseline_y] * len(line.get_xdata()))  # Use len() instead of .size
        fig.canvas.draw_idle()

    baseline_slider.on_changed(update)
    # Access the current figure's Tkinter window
    # canvas = plt.gcf().canvas
    # tk_window = canvas.manager.window
    # # Set the window size and position using Tkinter's geometry method
    # tk_window.geometry("960x540+0+0")
    plt.show()

    return int(baseline_slider.val)

def select_vertical_lines(image):
    '''
    Allow the user to select two vertical lines around the needle using sliders.
    Args:
        image: The input image for vertical line selection.
    '''
    global vertical_lines

    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Adjust the vertical lines using the sliders')

    # Initial positions for the vertical lines (starting at one-third and two-thirds of image width)
    init_left_line = image.shape[1] // 3
    init_right_line = 2 * image.shape[1] // 3

    # Create sliders for vertical lines
    ax_left_line = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    left_slider = Slider(ax_left_line, 'Left Line X', 0, image.shape[1], valinit=init_left_line)
    ax_right_line = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    right_slider = Slider(ax_right_line, 'Right Line X', 0, image.shape[1], valinit=init_right_line)

    # Create vertical lines based on the sliders
    left_line = ax.axvline(x=init_left_line, color='g', linestyle='--')
    right_line = ax.axvline(x=init_right_line, color='g', linestyle='--')

    def update_lines(val):
        left_line.set_xdata([left_slider.val] * len(left_line.get_ydata()))
        right_line.set_xdata([right_slider.val] * len(right_line.get_ydata()))
        fig.canvas.draw_idle()

    left_slider.on_changed(update_lines)
    right_slider.on_changed(update_lines)
    # Access the current figure's Tkinter window
    # canvas = plt.gcf().canvas
    # tk_window = canvas.manager.window
    # # Set the window size and position using Tkinter's geometry method
    # tk_window.geometry("960x540+0+0")
    plt.show()

    vertical_lines[0] = int(left_slider.val)
    vertical_lines[1] = int(right_slider.val)

    return vertical_lines

def average_y_for_same_x(points):
    '''
    Averages the y-coordinates for all points sharing the same x-coordinate.

    Args:
        points: List or array of [x, y] points.

    Returns:
        Numpy array of unique x-coordinates with averaged y-coordinates.
    '''
    if not points.any():
        raise ValueError("Input points array of points with same x-coordinate is empty.")
    
    x_dict = defaultdict(list)

    for point in points:
        x_dict[point[0]].append(point[1])

    averaged_points = []
    for x, y_values in x_dict.items():
        avg_y = np.mean(y_values)
        averaged_points.append([x, avg_y])

    return np.array(averaged_points)

def calculate_contact_angle(image, baseline_y, vertical_lines, 
                             prev_left_contact=None, prev_right_contact=None,
                             max_displacement=3.0):
    '''
    Calculates the static contact angle of a droplet on a surface based on the contour.

    Args:
        image: Input image of the droplet.
        baseline_y: Baseline y-coordinate for contact angle calculation.
        vertical_lines: Vertical mask boundaries
        prev_left_contact: Previous left contact point for temporal continuity
        prev_right_contact: Previous right contact point for temporal continuity
        max_displacement: Maximum allowed displacement between frames (pixels)

    Returns:
        Dictionary with calculation results or None if calculation fails.
    '''
    if image is None or baseline_y < 0 or baseline_y >= image.shape[0]:
        raise ValueError("Invalid image or baseline position.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lower_threshold, upper_threshold = read_last_thresholds()
    if lower_threshold is not None and upper_threshold is not None:
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    else:
        edges = cv2.Canny(blurred, 50, 150)
    # Create a mask to exclude regions between vertical lines
    mask = np.ones_like(blurred, dtype=np.uint8) * 255
    # Vertical line masking to exclude needle region
    mask[:, vertical_lines[0]:vertical_lines[1]] = 0  # Set the area between the lines to 0

    edges = edges * mask  # Apply mask to the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        print("No contours found!")
        return None

    all_points = np.vstack(contours)  # Stack all contour points into one array
    contour = all_points.reshape(-1, 2)
    contour_points = np.squeeze(contour)
    above_baseline_points = contour_points[contour_points[:, 1] < baseline_y]

    if len(above_baseline_points) < 3:
        print("Not enough points found above the baseline.")
        return None

    sorted_points = above_baseline_points[np.argsort(above_baseline_points[:, 0])]
    
    # Apply ellipse preprocessing to remove internal points
    all_points, ellipse_params = preprocess_contour_points(sorted_points, baseline_y)
    
    # Split points into left and right based on midpoint
    mid_x = np.mean(all_points[:, 0])
    left_points = all_points[all_points[:, 0] < mid_x]
    right_points = all_points[all_points[:, 0] >= mid_x]
    
    # Filter points to upper half based on y-coordinate range
    # For left points
    if len(left_points) > 0:
        y_min_left = np.min(left_points[:, 1])
        y_max_left = np.max(left_points[:, 1])
        range_y_left = y_max_left - y_min_left
        y_threshold_left = y_min_left + range_y_left / 2
        left_points = left_points[(left_points[:, 1] >= y_threshold_left) & 
                                  (left_points[:, 1] <= y_max_left)]
    
    # For right points
    if len(right_points) > 0:
        y_min_right = np.min(right_points[:, 1])
        y_max_right = np.max(right_points[:, 1])
        range_y_right = y_max_right - y_min_right
        y_threshold_right = y_min_right + range_y_right / 2
        right_points = right_points[(right_points[:, 1] >= y_threshold_right) & 
                                    (right_points[:, 1] <= y_max_right)]
    
    # Cap points to N closest to baseline (INCREASED from 20 to 100 for stability)
    max_points_for_fitting = 80  # CRITICAL: Increased for stable fits
    
    if len(left_points) > max_points_for_fitting:
        # Sort by y-coordinate in descending order (closest to baseline first)
        sorted_indices = np.argsort(left_points[:, 1])[::-1]
        left_points = left_points[sorted_indices[:max_points_for_fitting]]
    
    if len(right_points) > max_points_for_fitting:
        # Sort by y-coordinate in descending order (closest to baseline first)
        sorted_indices = np.argsort(right_points[:, 1])[::-1]
        right_points = right_points[sorted_indices[:max_points_for_fitting]]
    
    # Get image width for validation
    image_width = image.shape[1]
    
    # Calculate angles using NEW temporal continuity circle fitting
    result = calculate_contact_angles_circle_fit(
        baseline_y, left_points, right_points,
        prev_left_contact=prev_left_contact,
        prev_right_contact=prev_right_contact,
        image_width=image_width,
        max_displacement=max_displacement
    )
    
    # Check if both fits are valid
    if not result['both_valid']:
        # Return None if either side failed - graceful rejection
        error_msgs = []
        if not result['left_result']['valid']:
            error_msgs.append(f"Left: {result['left_result']['error']}")
        if not result['right_result']['valid']:
            error_msgs.append(f"Right: {result['right_result']['error']}")
        print(f"Frame rejected: {'; '.join(error_msgs)}")
        return None
    
    # Extract results
    left_result = result['left_result']
    right_result = result['right_result']
    
    left_angle = left_result['angle']
    right_angle = right_result['angle']
    left_contact_point = left_result['contact_point']
    right_contact_point = right_result['contact_point']
    left_fit = left_result['circle_params']
    right_fit = right_result['circle_params']
    
    # Calculate width and average angle
    width = np.linalg.norm(np.array(right_contact_point) - np.array(left_contact_point))
    avg_contact_angle = (left_angle + right_angle) / 2

    return {
        'left_angle': left_angle,
        'right_angle': right_angle,
        'avg_angle': avg_contact_angle,
        'width': width,
        'sorted_points': sorted_points,
        'left_points': left_points,
        'right_points': right_points,
        'baseline_y': baseline_y,
        'left_contact': left_contact_point,
        'right_contact': right_contact_point,
        'left_fit': left_fit,
        'right_fit': right_fit,
        'left_result': left_result,  # For tangent visualization
        'right_result': right_result,  # For tangent visualization
        'ellipse_params': ellipse_params  # For ellipse visualization
    }




def detect_hysteresis_points(widths, left_contact_angles, right_contact_angles):
    """
    Detect advancing and receding contact angles based on changes in width.

    Args:
        widths (list): List of width values.
        left_contact_angles (list): List of left contact angles.
        right_contact_angles (list): List of right contact angles.

    Returns:
        dict: Dictionary containing advancing and receding contact angles.
    """
    # Use peak detection to find advancing and receding points
    peaks, _ = scipy.signal.find_peaks(widths)  # Advancing (expansion)
    troughs, _ = scipy.signal.find_peaks(-np.array(widths))  # Receding (shrinkage)

    advancing_angles = [(left_contact_angles[i], right_contact_angles[i]) for i in peaks]
    receding_angles = [(left_contact_angles[i], right_contact_angles[i]) for i in troughs]

    return {
        "advancing_angles": advancing_angles,
        "receding_angles": receding_angles
    }

# Initialize video input and processing logic
def process_video_for_hysteresis(video_path):
    """
    Process video for hysteresis analysis, calculating contact angles and hysteresis.

    Args:
        video_path (str): Path to the input video file.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError("Error: Unable to open video file.")

    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Error: Could not read the first frame.")

    roi, scale = get_cropped_image(first_frame)
    if roi is None:
        raise ValueError("Failed to select ROI.")
    get_threshes(roi)
    baseline = select_baseline(roi)
    vertical_lines = select_vertical_lines(roi)
    if len(ref_point) == 2:
        x0, y0 = int(ref_point[0][0] / scale), int(ref_point[0][1] / scale)
        x1, y1 = int(ref_point[1][0] / scale), int(ref_point[1][1] / scale)
        # Get the coordinates in correct order
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        
    height, widthg, channels = roi.shape
    
    # Initialize video writer (you can adjust the output path, codec, and FPS as needed)
    output_path = r'/Users/shyamkaushik/Downloads/output.avi'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    output_video = cv2.VideoWriter(output_path, fourcc, 30.0, (widthg,height))  # Adjust size accordingly


    # Initialize matplotlib for live graph
    plt.ion()
    fig, ax = plt.subplots()  # 6.4 * 100 = 640, 4.8 * 100 = 480
    contact_angles = []
    left_contact_angles = []
    right_contact_angles = []
    frame_indices = []
    widths = []
    step = []
    line_contact_angle, = ax.plot([], [], label='Contact Angle (\xb0)', color='r')
    line_left_contact_angle, = ax.plot([], [], label='Left Contact Angle (\xb0)', color='g')
    line_right_contact_angle, = ax.plot([], [], label='Right Contact Angle (\xb0)', color='b')
    ax.set_xlim(0, 100)  # Initial limits, will be updated dynamically
    ax.set_ylim(90, 110)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Contact Angle (\xb0)")
    ax.legend()
      # Access the current figure's Tkinter window
    # canvas = plt.gcf().canvas
    # tk_window = canvas.manager.window
    # # Set the window size and position using Tkinter's geometry method
    # tk_window.geometry("960x540+0+0")  # Position at (0, 0) with size 960x540
    
    plot_path = r'/Users/shyamkaushik/Downloads/dynamic_plot.avi'
    frame_size = fig.canvas.get_width_height()
    output_plot = cv2.VideoWriter(plot_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, frame_size)
    
    frame_index = 0
    
    # Track previous contact points for temporal continuity
    prev_left_contact = None
    prev_right_contact = None
    
    # Track last valid values for graceful failure handling
    last_valid_left_ca = None
    last_valid_right_ca = None
    last_valid_avg_ca = None
    last_valid_width = None
    
    with open('contact_angle_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Left Contact Angle", "Right Contact Angle", "Average Contact Angle", "Width", "Status"])

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("End of video stream or error in reading the video file.")
                break
            
            # Process the frame for contact angle calculations with temporal continuity
            if cropping:
                frame = frame[y0:y1, x0:x1]
            
            result = calculate_contact_angle(
                frame, baseline, vertical_lines,
                prev_left_contact=prev_left_contact,
                prev_right_contact=prev_right_contact,
                max_displacement=float('inf')  # Disabled for hysteresis (large physical movements)
            )
            
            if result is None:
                # GRACEFUL FAILURE: Frame rejected, use previous valid values
                print(f"Frame {frame_index} rejected - holding previous contact points")
                if last_valid_left_ca is not None:
                    # Use last valid values but mark as "held"
                    contact_angles.append(last_valid_avg_ca)
                    left_contact_angles.append(last_valid_left_ca)
                    right_contact_angles.append(last_valid_right_ca)
                    widths.append(last_valid_width)
                    writer.writerow([frame_index, last_valid_left_ca, last_valid_right_ca, 
                                   last_valid_avg_ca, last_valid_width, "HELD"])
                    frame_indices.append(frame_index)
                # Do NOT update prev_left_contact or prev_right_contact - hold them
                frame_index += 1
                continue

            # Extract results from dictionary
            left_ca = result['left_angle']
            right_ca = result['right_angle']
            contact_angle = result['avg_angle']
            width = result['width']
            sorted_points = result['sorted_points']
            left_points = result['left_points']
            right_points = result['right_points']
            baseline_y = result['baseline_y']
            left_intersection = result['left_contact']
            right_intersection = result['right_contact']
            left_fit = result['left_fit']
            right_fit = result['right_fit']
            left_result = result['left_result']  # For tangent visualization
            right_result = result['right_result']  # For tangent visualization
            ellipse_params = result['ellipse_params']  # For ellipse visualization

            
            if contact_angle is not None and contact_angle <180:
                contact_angles.append(contact_angle)
                left_contact_angles.append(left_ca)
                right_contact_angles.append(right_ca)
                widths.append(width)
                writer.writerow([frame_index, left_ca, right_ca, contact_angle, width, "VALID"])
                frame_indices.append(frame_index)
                
                # Update previous contact points for next frame
                prev_left_contact = left_intersection
                prev_right_contact = right_intersection
                
                # Update last valid values
                last_valid_left_ca = left_ca
                last_valid_right_ca = right_ca
                last_valid_avg_ca = contact_angle
                last_valid_width = width

            filtered_contact_angles = [angle for angle in contact_angles if angle is not None]


            # Update live plot for all three lines
            line_contact_angle.set_xdata(frame_indices)
            line_contact_angle.set_ydata(contact_angles)

            line_left_contact_angle.set_xdata(frame_indices)
            line_left_contact_angle.set_ydata(left_contact_angles)

            line_right_contact_angle.set_xdata(frame_indices)
            line_right_contact_angle.set_ydata(right_contact_angles)

            ax.set_xlim(0, max(100, frame_index))
            ax.set_ylim(0, 180)
            plt.draw()
            plt.pause(0.01)
            
            # Inside your existing loop where you update the plot
            fig.canvas.draw() 
            frame_p = np.array(fig.canvas.renderer.buffer_rgba())
            frame_p = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  
            output_plot.write(frame_p)

            # Function to draw tangent lines for both circle and line fits
            def draw_tangent_line_from_result(result_dict, side):
                """Draw tangent line from either circle fit or line fit result"""
                if result_dict is None or not result_dict.get('valid', False):
                    return [], []
                
                intersection = result_dict['contact_point']
                
                # Check if this is a line fit
                if result_dict.get('fit_type') == 'line':
                    # For line fit, use the slope directly
                    slope = result_dict.get('tangent_slope')
                    
                    # Handle vertical line (infinite slope)
                    if slope == float('inf') or slope == float('-inf') or abs(slope) > 1000:
                        # Vertical line
                        line_length = 100
                        tangent_line_x = [intersection[0], intersection[0]]
                        tangent_line_y = [intersection[1] - line_length, intersection[1] + line_length]
                        return tangent_line_x, tangent_line_y
                    
                    # Create line from slope: y = mx + b
                    # The tangent IS the fitted line
                    line_length = 100
                    dx = line_length
                    dy = slope * dx
                    
                    tangent_line_x = [intersection[0] - dx, intersection[0] + dx]
                    tangent_line_y = [intersection[1] - dy, intersection[1] + dy]
                    
                    return tangent_line_x, tangent_line_y
                
                # For circle fit
                circle_params = result_dict.get('circle_params')
                if circle_params is None:
                    return [], []
                
                x0, y0, r = circle_params
                dx = intersection[0] - x0
                dy = intersection[1] - y0
                
                # Tangent vector perpendicular to radius
                if side == 'left':
                    tangent_x = -dy
                    tangent_y = dx
                else:
                    tangent_x = dy
                    tangent_y = -dx
                
                # Normalize tangent vector
                tangent_length = np.sqrt(tangent_x**2 + tangent_y**2)
                if tangent_length > 0:
                    tangent_x /= tangent_length
                    tangent_y /= tangent_length
                
                # Create tangent line points
                line_length = 100
                tangent_line_x = [intersection[0] - line_length * tangent_x, 
                                  intersection[0] + line_length * tangent_x]
                tangent_line_y = [intersection[1] - line_length * tangent_y, 
                                  intersection[1] + line_length * tangent_y]
                
                return tangent_line_x, tangent_line_y

            
            # Generate tangent lines from circle/line fits using result dictionaries
            x_tangent_left, y_tangent_left = draw_tangent_line_from_result(left_result, 'left')
            x_tangent_right, y_tangent_right = draw_tangent_line_from_result(right_result, 'right')

            
            # Draw baseline
            cv2.line(frame, (0, int(baseline_y)), (frame.shape[1], int(baseline_y)), (255, 255, 0), 1, cv2.LINE_AA)
            
            # Draw ellipse used for preprocessing (DISABLED)
            # if ellipse_params is not None:
            #     center = ellipse_params['center']
            #     semi_major = ellipse_params['semi_major']
            #     semi_minor = ellipse_params['semi_minor']
            #     # cv2.ellipse expects center as tuple of ints, axes as tuple, angle, start/end angles, color, thickness
            #     cv2.ellipse(frame, 
            #                (int(center[0]), int(center[1])),  # center
            #                (int(semi_major), int(semi_minor)),  # axes (horizontal, vertical)
            #                0,  # angle of rotation
            #                0, 360,  # start and end angles
            #                (255, 0, 255), 1)  # magenta color, thickness 1
            
            # Draw fitted circles
            if left_fit is not None:
                x0_left, y0_left, r_left = left_fit
                cv2.circle(frame, (int(x0_left), int(y0_left)), int(r_left), (144, 238, 144), 1)  # Light green circle for left
            
            if right_fit is not None:
                x0_right, y0_right, r_right = right_fit
                cv2.circle(frame, (int(x0_right), int(y0_right)), int(r_right), (144, 238, 144), 1)  # Light green circle for right

            # Draw tangent lines on the frame
            if len(x_tangent_left) >= 2:
                cv2.line(frame, 
                        (int(x_tangent_left[0]), int(y_tangent_left[0])), 
                        (int(x_tangent_left[1]), int(y_tangent_left[1])), 
                        (255, 255, 0), 1)  # Cyan tangent for left

            if len(x_tangent_right) >= 2:
                cv2.line(frame, 
                        (int(x_tangent_right[0]), int(y_tangent_right[0])), 
                        (int(x_tangent_right[1]), int(y_tangent_right[1])), 
                        (255, 255, 0), 1)  # Cyan tangent for right
            # Show the annotated frame
            cv2.imshow("Video Analysis", frame)

            output_video.write(frame)
            frame_index += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    advancing_ca, receding_ca = None, None
    for i in range(1, len(widths)):
        step.append(widths[i] - widths[i - 1])
    for i in range(len(step)):
        if step[i] == max(step):
            advancing_ca = contact_angles[i]
        if step[i] == min(step) and step[i]<0:
            receding_ca = contact_angles[i]
    if advancing_ca is None:
        raise ValueError("Failed to calculate the advancing contact angle. Ensure the needle method algorithm is correctly implemented and that the video provides clear boundary motion.")
    if receding_ca is None:
        raise ValueError("Failed to calculate the receding contact angle. Ensure the needle method algorithm is correctly implemented and that the video provides clear boundary motion.")
    Contact_Angle_Hysteresis = advancing_ca - receding_ca

    text_content = (
        f"Advancing Contact Angle: {advancing_ca}\n"
        f"Receding Contact Angle: {receding_ca}\n"
        f"Contact Angle Hysteresis: {Contact_Angle_Hysteresis}\n"
    )

    file_path = "Contact_Angle_Hysteresis.txt"
    with open(file_path, "w") as file:
        file.write(text_content)

    cap.release()
    output_plot.release()
    output_video.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()


def hysteresis():
    # video_retrieval_from_phone()
    # os.chdir(r"C:\Users\91982\OneDrive\Desktop\SURA")
    # video_path= str(load_latest_image_path())
    # video_path = r"C:\Users\91982\Videos\hysteresis.mp4"
    video_path=r"/Users/shyamkaushik/Desktop/Golden Days/Gonio/Inputs/Dynamic Contact Angle Input Videos/hys.mp4"
    process_video_for_hysteresis(video_path)

hysteresis()

# if __name__ == "__main__":
#     video_path = r"C:\Users\91982\Videos\hysteresis.mp4"  # Replace with the path to your video file
#     process_video_for_hysteresis(video_path)
