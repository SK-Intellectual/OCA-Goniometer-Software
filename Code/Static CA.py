import cv2
import math
import os
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from collections import defaultdict
from ADB_extractor import image_retrieval_from_phone
from Thresholds import get_threshes
from scipy.stats import linregress
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.optimize import least_squares

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
    # Initial guess for circle center (mean of points) and radius
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])
    
    # Calculate initial radius estimate
    distances = np.sqrt((points[:, 0] - x_mean)**2 + (points[:, 1] - y_mean)**2)
    r_initial = np.mean(distances)
    
    # Compute weights if baseline is provided
    weights = None
    if baseline_y is not None:
        # Distance from baseline (vertical distance)
        distance_from_baseline = np.abs(points[:, 1] - baseline_y)
        
        # Exponential decay: points near baseline get weight ~1, far points get weight ~0
        # Use characteristic decay length = 50 pixels
        decay_factor = 50.0
        weights = np.exp(-distance_from_baseline / decay_factor)
        
        # Normalize weights so they sum to number of points (maintains scale)
        weights = weights * len(points) / np.sum(weights)
    
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

def fit_line_and_calculate_angle_pca(points, baseline_y, side):
    """
    Fallback function for collinear points: fit a line instead of circle.
    Uses PCA-based Total Least Squares to handle vertical/near-vertical lines.
    
    Args:
        points: Array of (x, y) coordinates
        baseline_y: Y-coordinate of baseline
        side: 'left' or 'right'
    
    Returns:
        tuple: (contact_angle, intersection_x, line_params) or None if failed
              where line_params is (slope, intercept, direction) for line fit
    """
    if len(points) < 2:
        print(f"[{side.upper()}] Not enough points for line fit")
        return None
    
    # PCA-based line fit (Total Least Squares)
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
    
    # Find intersection with baseline
    if abs(direction[1]) < 1e-10:
        print(f"[{side.upper()}] Line is horizontal, no baseline intersection")
        return None
    
    t = (baseline_y - mean[1]) / direction[1]
    intersection_x = mean[0] + t * direction[0]
    
    # Calculate contact angle from line direction
    if abs(direction[0]) < 1e-10:
        # Vertical line → 90 degrees
        contact_angle = 90.0
        slope = float('inf')
    else:
        slope = direction[1] / direction[0]
        angle_rad = math.atan(slope)
        angle_deg = math.degrees(angle_rad)
        
        # Determine if obtuse based on slope direction
        if side == 'left':
            if slope < 0:
                contact_angle = abs(angle_deg)
            else:
                contact_angle = 180 - abs(angle_deg)
        else:  # right side
            if slope > 0:
                contact_angle = abs(angle_deg)
            else:
                contact_angle = 180 - abs(angle_deg)
        
        contact_angle = max(0, min(180, contact_angle))
    
    # Calculate intercept: y = mx + b => b = y - mx
    # Use mean point for calculation
    if abs(direction[0]) < 1e-10:
        intercept = None  # Vertical line, use vertical line representation
    else:
        intercept = mean[1] - slope * mean[0]
    
    # Store line parameters for tangent drawing: (slope, intercept, direction_vector)
    line_params = (slope, intercept, direction)
    
    print(f"[{side.upper()}] Using LINE FIT (PCA): angle={contact_angle:.2f}°")
    return contact_angle, intersection_x, line_params


def circle_fit_errors(points, x0, y0, r, eps=1e-12):
    """
    Compute geometric error metrics for a fitted circle.

    points : (N, 2) array
    x0, y0 : circle center
    r      : circle radius
    """
    points = np.asarray(points)
    
    # Distances of points from center
    distances_from_center = np.sqrt((points[:, 0] - x0)**2 +
                                    (points[:, 1] - y0)**2)
    
    # Radial residuals: how far each point is from the circle radius
    radial_residuals = distances_from_center - r
    
    # RMSE of radial residuals (primary metric)
    rmse_radial = np.sqrt(np.mean(radial_residuals**2))
    
    # Relative RMSE (dimensionless, fraction of radius)
    if abs(r) > eps:
        rel_rmse = rmse_radial / abs(r)
    else:
        rel_rmse = np.nan  # radius ~ 0 ⇒ relative error meaningless
    
    # Max absolute radial error (optional, for robustness checks)
    max_abs_error = np.max(np.abs(radial_residuals))
    
    return {
        "rmse_radial": rmse_radial,
        "rel_rmse": rel_rmse,
        "max_abs_error": max_abs_error,
    }


def calculate_contact_angle_from_circle(baseline_y, points, side='left', all_points_x_range=None):
    """
    Calculates the contact angle by fitting points to a circle and computing
    the tangent angle at the intersection point with the baseline.
    
    The contact angle is measured from the horizontal baseline to the tangent
    of the droplet surface at the contact point, measured through the liquid phase.
    
    Args:
        baseline_y (float): Y-coordinate of the baseline.
        points (np.array): Array of shape (n, 2) containing [x, y] coordinates of the droplet edge.
        side (str): 'left' or 'right' - which side of the droplet.
        all_points_x_range (tuple): (x_min, x_max) of the entire droplet for outer fit detection.
    
    Returns:
        tuple: (contact_angle, intersection_x, (x0, y0, r), is_outer_fit)
    """
    # Fit circle to points with weighted fitting (prioritize points near baseline)
    fit_result = fit_circle_to_points(points, baseline_y=baseline_y)
    
    if fit_result is None:
        print(f"Failed to fit circle for {side} side.")
        return None
    
    x0, y0, r = fit_result
    
    # COLLINEARITY DETECTION: If radius is too large, points are nearly collinear
    # Fall back to line fitting instead of using degenerate circle
    COLLINEARITY_THRESHOLD = 5000  # pixels
    
    if r > COLLINEARITY_THRESHOLD:
        print(f"[{side.upper()}] Circle radius {r:.1f}px > {COLLINEARITY_THRESHOLD}px, falling back to LINE FIT")
        line_result = fit_line_and_calculate_angle_pca(points, baseline_y, side)
        if line_result is not None:
            contact_angle, intersection_x, line_params = line_result
            # Return with line_params marked as line fit (negative radius to indicate line fit)
            return contact_angle, intersection_x[0] if isinstance(intersection_x, (list, np.ndarray)) else intersection_x, ('line', line_params), False
        else:
            print(f"[{side.upper()}] Line fit also failed, using circle fit anyway")
            # Continue with circle fit even though it's degenerate
    
    dy = baseline_y - y0
    
    # Check if circle intersects baseline
    if abs(dy) > r:
        print(f"[{side.upper()}] Warning: Circle does not intersect baseline (|dy|={abs(dy):.1f} > r={r:.1f})")
        # Use closest point on circle to baseline as a fallback
        # Instead of calculating intersection, return None to indicate failure
        return None
    
    # Calculate the horizontal distance from x0 to get TWO possible intersection points
    dx = np.sqrt(r**2 - dy**2)
    intersection_left = [x0 - dx, baseline_y]   # Left intersection point
    intersection_right = [x0 + dx, baseline_y]  # Right intersection point
    
    # ============================================================================
    # OUTER CIRCLE FIT DETECTION
    # ============================================================================
    # Check if circle center is within droplet's x-range
    # If outside → outer fit → select opposite intersection point
    
    is_outer_fit = False
    
    if all_points_x_range is not None:
        x_min, x_max = all_points_x_range
        center_inside_droplet = (x_min <= x0 <= x_max)
        
        if not center_inside_droplet:
            is_outer_fit = True
            print(f"[{side.upper()}] Outer circle fit detected: center x={x0:.1f} outside range [{x_min:.1f}, {x_max:.1f}]")
    
    # Select correct intersection point based on fit type
    if is_outer_fit:
        # OUTER FIT: Select opposite intersection
        if side == 'left':
            intersection_point = intersection_right  # For left side outer fit, use RIGHT intersection
            print(f"[{side.upper()}] Using RIGHT intersection point for outer fit: x={intersection_right[0]:.1f}")
        else:  # right side
            intersection_point = intersection_left   # For right side outer fit, use LEFT intersection
            print(f"[{side.upper()}] Using LEFT intersection point for outer fit: x={intersection_left[0]:.1f}")
    else:
        # INNER FIT: Use normal selection (original logic)
        if side == 'left':
            intersection_point = intersection_left
        else:
            intersection_point = intersection_right
    
    # Vector from circle center to intersection point (radius vector)
    dx = intersection_point[0] - x0
    dy = intersection_point[1] - y0
    
    # The tangent to the circle at the intersection point is perpendicular to the radius
    # The tangent vector is obtained by rotating the radius vector by 90 degrees
    # Tangent vector: (-dy, dx) or (dy, -dx) depending on orientation
    
    # Determine which tangent direction to use based on which side
    # We want the tangent pointing in the direction of the droplet surface
    
    # Check if the center is above or below the intersection point
    center_above = (y0 < intersection_point[1])  # y increases downward in images
    
    # Determine the tangent vector
    # The tangent should point in the direction along the droplet surface
    if side == 'left':
        # For left side, positive tangent direction should go left and upward (for acute)
        # or left and downward (for obtuse)
        tangent_x = -dy
        tangent_y = dx
    else:  # right side
        # For right side, positive tangent direction should go right and upward (for acute)
        # or right and downward (for obtuse)
        tangent_x = dy
        tangent_y = -dx
    
    # Calculate the angle of the tangent vector with respect to horizontal (baseline)
    # atan2 gives angle from positive x-axis, counterclockwise
    tangent_angle_rad = math.atan2(tangent_y, tangent_x)
    
    # The slope from the angle
    if abs(math.cos(tangent_angle_rad)) > 1e-10:
        tangent_slope = math.tan(tangent_angle_rad)
    else:
        tangent_slope = float('inf')  # Vertical tangent
    
    # Calculate contact angle from the tangent slope
    # Contact angle is measured from baseline (horizontal) through the liquid
    
    # ============================================================================
    # IMPROVED: Robust Acute/Obtuse Detection Using Circle Center Position
    # ============================================================================
    # PRIMARY METHOD: Use geometric relationship between circle center and baseline
    # For a sessile drop on a horizontal surface:
    #   - If circle center is ABOVE baseline (y < baseline_y), angle is OBTUSE (>90°)
    #   - If circle center is BELOW baseline (y > baseline_y), angle is ACUTE (<90°)
    # This is geometrically sound and works reliably even near 90°
    
    # EXCEPTION: Outer circle fits are ALWAYS acute regardless of center position
    
    # Use the geometric method (circle center position relative to baseline)
    center_above_baseline = (y0 < baseline_y)  # y increases downward in images
    
    # For outer fits, override detection - they are always acute
    if is_outer_fit:
        is_obtuse = False  # Outer fits are always acute
        print(f"[{side.upper()}] Outer fit: forcing angle to be acute")
    else:
        is_obtuse = center_above_baseline
    
    # Calculate the contact angle
    # The tangent slope determines the angle
    angle_from_horizontal = math.degrees(math.atan(tangent_slope))
    
    # Adjust based on side and obtuseness
    if side == 'left':
        if is_obtuse:
            # Obtuse left angle: 90° < θ < 180°
            contact_angle = 180 + angle_from_horizontal if angle_from_horizontal < 0 else 180 - angle_from_horizontal
        else:
            # Acute left angle: 0° < θ < 90°
            contact_angle = -angle_from_horizontal if angle_from_horizontal < 0 else angle_from_horizontal
            
    else:  # right side
        if is_obtuse:
            # Obtuse right angle: 90° < θ < 180°
            contact_angle = 180 - abs(angle_from_horizontal)
        else:
            # Acute right angle: 0° < θ < 90°
            contact_angle = abs(angle_from_horizontal)
    
    # For outer fits, if calculated angle is obtuse, convert to acute
    if is_outer_fit and contact_angle > 90:
        contact_angle = 180 - contact_angle
        print(f"[{side.upper()}] Outer fit angle adjustment: {contact_angle + (180 - 2*contact_angle):.2f}° → {contact_angle:.2f}°")
    
    # Ensure angle is in valid range [0, 180]
    contact_angle = max(0, min(180, contact_angle))
    
    # Return angle, intersection x-coordinate, fitted circle parameters, and fit type
    return contact_angle, intersection_point[0], (x0, y0, r), is_outer_fit


def calculate_contact_angles_circle_fit(baseline_y, left_points, right_points, all_points_x_range=None):
    """
    Main function to calculate both left and right contact angles using circle fitting.
    
    Args:
        baseline_y (float): Y-coordinate of the baseline.
        left_points (np.array): Points on the left side of the droplet, shape (n, 2).
        right_points (np.array): Points on the right side of the droplet, shape (n, 2).
        all_points_x_range (tuple): (x_min, x_max) of entire droplet for outer fit detection.
    
    Returns:
        tuple: (left_angle, right_angle, left_intersection, right_intersection, left_fit, right_fit)
               where left_fit and right_fit are tuples (x0, y0, r) for circle parameters.
    """
    left_result = calculate_contact_angle_from_circle(
        baseline_y, left_points, side='left', all_points_x_range=all_points_x_range)
    right_result = calculate_contact_angle_from_circle(
        baseline_y, right_points, side='right', all_points_x_range=all_points_x_range)
    
    # Handle None returns (circle doesn't intersect baseline)
    if left_result is None:
        print("[LEFT] Circle fit failed, using default values")
        left_angle, left_intersection, left_fit, left_is_outer = 90.0, 0, None, False
    else:
        left_angle, left_intersection, left_fit, left_is_outer = left_result
        
    if right_result is None:
        print("[RIGHT] Circle fit failed, using default values")
        right_angle, right_intersection, right_fit, right_is_outer = 90.0, 0, None, False
    else:
        right_angle, right_intersection, right_fit, right_is_outer = right_result
    
    return left_angle, right_angle, left_intersection, right_intersection, left_fit, right_fit


def preprocess_contour_points(points, baseline_y):
    """
    Preprocesses contour points by removing internal points using an ellipse filter.
    
    Creates an ellipse inside the droplet and removes points within it, keeping only
    boundary points. The ellipse is positioned:
    - Horizontally: 1/10th of width inward from left/right extremes
    - Vertically: Centered at baseline level
    
    Args:
        points (np.array): Array of shape (n, 2) containing [x, y] coordinates.
        baseline_y (float): Y-coordinate of the baseline (contact line).
    
    Returns:
        tuple: (filtered_points, ellipse_params)
            - filtered_points: Array containing only boundary points (outside the ellipse)
            - ellipse_params: Dict with 'center', 'semi_major', 'semi_minor' for visualization
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
    
    # Find baseline-contour intersection points (contact points)
    # These are the points where the droplet actually touches the surface
    baseline_tolerance = 2.0  # pixels - tolerance for baseline proximity
    near_baseline = points[np.abs(points[:, 1] - baseline_y) < baseline_tolerance]
    
    if len(near_baseline) < 2:
        # Fallback: use leftmost/rightmost if no baseline intersection found
        print("Warning: Could not find baseline intersection points, using leftmost/rightmost as fallback")
        left_contact_x = leftmost_point[0]
        right_contact_x = rightmost_point[0]
    else:
        # Left contact: minimum x among baseline points
        left_contact_x = np.min(near_baseline[:, 0])
        # Right contact: maximum x among baseline points
        right_contact_x = np.max(near_baseline[:, 0])
    
    # Calculate contact width (distance between contact points)
    contact_width = right_contact_x - left_contact_x
    
    # Calculate ellipse endpoints INSIDE the contact points
    # Move inward by 1/10th of the contact width from each contact point
    left_ellipse_x = left_contact_x + contact_width / 10
    right_ellipse_x = right_contact_x - contact_width / 10
    
    # Ellipse center (horizontal midpoint, vertical at baseline)
    ellipse_center_x = (left_ellipse_x + right_ellipse_x) / 2
    ellipse_center_y = baseline_y  # Center on the baseline (contact line)
    
    # Semi-major axis (horizontal): half the distance between ellipse endpoints
    semi_major_a = (right_ellipse_x - left_ellipse_x) / 2
    
    # Calculate vertical distance for ellipse top
    # Top of ellipse: 1/8th distance lower than topmost point
    height_from_top = topmost_point[1] - ellipse_center_y  # Negative if top is above center
    top_ellipse_y = topmost_point[1] + abs(height_from_top) / 8
    
    # Semi-minor axis (vertical): distance from center to top
    semi_minor_b = abs(ellipse_center_y - top_ellipse_y)
    
    # Filter points: keep only those OUTSIDE the ellipse
    # Ellipse equation: ((x - cx)/a)^2 + ((y - cy)/b)^2 <= 1 (inside)
    # We keep points where this is > 1 (outside)
    
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



def calculate_r_squared(y_true, y_pred):
    """
    Calculates the R-squared value for a given fit.
    
    Args:
        y_true (np.array): True y-coordinates.
        y_pred (np.array): Predicted y-coordinates from the model.
    
    Returns:
        float: R-squared value.
    """
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    if ss_total == 0:
        return 0
    return 1 - (ss_residual / ss_total)

def filter_points_for_fitting(all_points, intersection_point_x, side, num_points=100):
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

# Global variables for cropping
ref_point = []
cropping = False

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
            # Resize the window (width, height)
            cv2.resizeWindow("Crop Image", 960, 540)
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
        cv2.resizeWindow("Crop Image", 960, 540)
        # Move the window (x, y position on screen)
        cv2.moveWindow("Crop Image", 0, 0)
        cv2.imshow("Crop Image", param)

# Function to resize the image to fit within the screen
def resize_to_fit_screen(image, screen_width=800, screen_height=600):
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
    global ref_point
    
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    
    # Resize the image to fit the screen
    resized_image, scale = resize_to_fit_screen(image)

    clone = resized_image.copy()

    # Set up the window and set the mouse callback function for cropping
    cv2.namedWindow("Crop Image")
    # Resize the window (width, height)
    cv2.resizeWindow("Crop Image", 960, 540)
    # Move the window (x, y position on screen)
    cv2.moveWindow("Crop Image", 0, 0)
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
            break

        # Press 'n' to skip cropping
        elif key == ord("n"):
            cv2.destroyAllWindows()  # Close the window
            return image  # Return the original image without cropping

        # Check if the window is closed
        if cv2.getWindowProperty("Crop Image", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed.")
            cv2.destroyAllWindows()
            return image  # Return the original image without cropping

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
        return cropped_image
    else:
        return image

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
    
    # Dictionary to store points grouped by x-coordinates
    x_dict = defaultdict(list)

    # Group points by their x-coordinate
    for point in points:
        x_dict[point[0]].append(point[1])  # Add y-coordinates for the same x-coordinate

    # Create a new list of points with unique x and averaged y
    averaged_points = []
    for x, y_values in x_dict.items():
        avg_y = np.mean(y_values)  # Calculate the average of y-values for the same x-coordinate
        averaged_points.append([x, avg_y])

    return np.array(averaged_points)

def calculate_contact_angle(image, baseline_y):
    '''
    Calculates the static contact angle of a droplet on a surface based on the contour.

    Args:
        image: Input image of the droplet.
        baseline_y: Baseline y-coordinate for contact angle calculation.

    Returns:
        Average contact angle or None if calculation fails.
    '''
    if image is None or baseline_y < 0 or baseline_y >= image.shape[0]:
        raise ValueError("Invalid image or baseline position.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Show images
    # cv2.imshow("Grayscale", gray)
    # cv2.imshow("Blurred", blurred)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Perform edge detection using Canny
    lower_threshold, upper_threshold = read_last_thresholds()
    if lower_threshold is not None and upper_threshold is not None:
        edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    else:
        edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        print("No contours found!")
        return None

    all_points = np.vstack(contours)  # Stack all contour points into one array
    contour = all_points.reshape(-1, 2)

    # Get the x, y coordinates of the contour points
    contour_points = np.squeeze(contour)

    # Filter contour points above the baseline
    above_baseline_points = contour_points[contour_points[:, 1] < baseline_y]

    if len(above_baseline_points) < 3:
        print("Not enough points found above the baseline for contact angle calculation.")
        return None

    # Sort points by x-coordinate
    sorted_points = above_baseline_points[np.argsort(above_baseline_points[:, 0])]
    all_points, ellipse_params = preprocess_contour_points(sorted_points, baseline_y)

    with open('ca.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        # Write all rows at once
        writer.writerows(sorted_points)
    file.close()
    # # Find intersection points
    # intersection_points = sorted_points[np.isclose(sorted_points[:, 1], baseline_y, atol=1.0)]
    # if len(intersection_points) < 2:
    #     print("Could not find sufficient intersection points with the baseline.")
    #     return None

    # left_intersection = intersection_points[0]
    # right_intersection = intersection_points[-1]

    # # Case 1: Intersection points are the first and last elements
    # if np.all(sorted_points[0] == left_intersection) and np.all(sorted_points[-1] == right_intersection):
    #     obtuse = False
    #     left_points = sorted_points[:501]  # First 11 points
    #     left_points = average_y_for_same_x(left_points)
    #     right_points = sorted_points[-501:]  # Last 11 points
    #     right_points  = average_y_for_same_x(right_points)

    # else:
    #     # Case 2: Intersection points are in between
    #     obtuse = True
    #     # Distance threshold from the baseline (y-axis proximity check)
    #     distance_threshold = 80  # Adjust based on your image scale

    #     # Calculate distance from baseline for all points
    #     distances_from_baseline = np.abs(sorted_points[:, 1] - baseline_y)

    #     # Select only points within the distance threshold near the left intersection
    #     left_points = sorted_points[(distances_from_baseline < distance_threshold) | 
    #                                 (sorted_points[:, 0] <= left_intersection[0])]
    #     left_points = average_y_for_same_x(left_points)
    #     # Select only points within the distance threshold near the right intersection
    #     right_points = sorted_points[(distances_from_baseline < distance_threshold) | 
    #                                 (sorted_points[:, 0] >= right_intersection[0])]
    #     right_points = average_y_for_same_x(right_points)

    # # Polynomial fitting and contact angle calculation
    # def calculate_angle(points, intersection_point):
    #     poly_coeffs = np.polyfit(points[:, 0], points[:, 1], 2)
    #     poly = np.poly1d(poly_coeffs)
    #     slope = np.polyder(poly)(intersection_point[0])
    #     angle = np.arctan(slope) * (180 / np.pi)
    #     return angle

    # # Calculate left and right contact angles
    # if obtuse:
    #     left_angle = 180 - calculate_angle(left_points, left_intersection)
    #     right_angle = 180 + calculate_angle(right_points, right_intersection)
    # else : 
    #     left_angle = -calculate_angle(left_points, left_intersection)
    #     right_angle = calculate_angle(right_points, right_intersection)
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
    
    # Calculate x-range from all_points for outer fit detection
    x_min = np.min(all_points[:, 0])
    x_max = np.max(all_points[:, 0])
    all_points_x_range = (x_min, x_max)
    
    # Cap points to N closest to baseline (matching hysteresis implementation)
    max_points_for_fitting = 80  # CRITICAL: Increased for stable fits
    
    if len(left_points) > max_points_for_fitting:
        # Sort by y-coordinate in descending order (closest to baseline first)
        sorted_indices = np.argsort(left_points[:, 1])[::-1]
        left_points = left_points[sorted_indices[:max_points_for_fitting]]
    
    if len(right_points) > max_points_for_fitting:
        # Sort by y-coordinate in descending order (closest to baseline first)
        sorted_indices = np.argsort(right_points[:, 1])[::-1]
        right_points = right_points[sorted_indices[:max_points_for_fitting]]
    
    
    # Calculate angles
    left_angle, right_angle, le, ri, left_fit, right_fit = calculate_contact_angles_circle_fit(
        
        baseline_y, left_points, right_points, all_points_x_range=all_points_x_range,
    )
    left_intersection = [le,baseline_y]
    right_intersection = [ri, baseline_y]
    
    
        
    # Use circle fits from angle calculation for visualization and error metrics
    # (No need to refit - we already have the circle parameters)
    
    # Initialize defaults
    x0_l, y0_l, r_l = 0, 0, 0
    x0_r, y0_r, r_r = 0, 0, 0
    left_errors = {"rmse_radial": 0.0, "rel_rmse": 0.0, "max_abs_error": 0.0}
    right_errors = {"rmse_radial": 0.0, "rel_rmse": 0.0, "max_abs_error": 0.0}
    
    # Calculate error metrics for goodness of fit using the same circle fits
    if left_fit is not None:
        # Check if it's a circle fit (not a line fit)
        if not (isinstance(left_fit, tuple) and len(left_fit) == 2 and left_fit[0] == 'line'):
            x0_l, y0_l, r_l = left_fit
            left_errors = circle_fit_errors(left_points, x0_l, y0_l, r_l)
    
    if right_fit is not None:
        # Check if it's a circle fit (not a line fit)
        if not (isinstance(right_fit, tuple) and len(right_fit) == 2 and right_fit[0] == 'line'):
            x0_r, y0_r, r_r = right_fit
            right_errors = circle_fit_errors(right_points, x0_r, y0_r, r_r)
    
    print(f"\nLeft Contact Angle: {left_angle:.2f}°")
    print(f"  RMSE Radial: {left_errors['rmse_radial']:.4f} px, Rel RMSE: {left_errors['rel_rmse']:.4f}, Max Error: {left_errors['max_abs_error']:.4f} px")
    print(f"Right Contact Angle: {right_angle:.2f}°")
    print(f"  RMSE Radial: {right_errors['rmse_radial']:.4f} px, Rel RMSE: {right_errors['rel_rmse']:.4f}, Max Error: {right_errors['max_abs_error']:.4f} px")
    avg_angle = (left_angle + right_angle) / 2
    print(f"Average Contact Angle: {avg_angle:.2f}°")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(image[:, :, ::-1], alpha=1)
    
    # Plot ellipse filter used for preprocessing
    if ellipse_params is not None:
        center_x, center_y = ellipse_params['center']
        semi_major = ellipse_params['semi_major']
        semi_minor = ellipse_params['semi_minor']
        
        # Create ellipse
        theta = np.linspace(0, 2*np.pi, 300)
        ellipse_x = center_x + semi_major * np.cos(theta)
        ellipse_y = center_y + semi_minor * np.sin(theta)
        
        # ax.plot(ellipse_x, ellipse_y, 'c--', linewidth=2, 
        #         label=f'Ellipse Filter (a={semi_major:.1f}, b={semi_minor:.1f})', alpha=0.5)
        # ax.plot(center_x, center_y, 'c+', markersize=10, markeredgewidth=2, 
        #         label='Ellipse Center', alpha=0.7)
        
        # Calculate and plot removed points (points inside ellipse)
        normalized_x = (sorted_points[:, 0] - center_x) / semi_major
        normalized_y = (sorted_points[:, 1] - center_y) / semi_minor
        inside_ellipse = (normalized_x**2 + normalized_y**2) <= 1.0
        removed_points = sorted_points[inside_ellipse]
        
        if len(removed_points) > 0:
            ax.plot(removed_points[:, 0], removed_points[:, 1], 'gray', 
                    marker='.', linestyle='', markersize=2, label='Removed Points', alpha=0.4)
    
    # Plot all original points
    ax.plot(all_points[:, 0], all_points[:, 1], 'k.', 
            markersize=2, label='All Contour Points', alpha=0.3)
    ax.plot(sorted_points[:, 0], sorted_points[:, 1], 'r.', 
            markersize=2, label='pre Contour Points', alpha=0.3)
    
    # Plot left and right fitted points with different colors
    ax.plot(left_points[:, 0], left_points[:, 1], 'bo', 
            markersize=4, label='Left Fitted Points', alpha=0.5)
    ax.plot(right_points[:, 0], right_points[:, 1], 'ro', 
            markersize=4, label='Right Fitted Points', alpha=0.5)
    
    # Plot intersection points as dots (not crosses)
    ax.plot(left_intersection[0], left_intersection[1], 'go', 
            markersize=8, label='Left Intersection')
    ax.plot(right_intersection[0], right_intersection[1], 'mo', 
            markersize=8, label='Right Intersection')
    
    # Draw fitted circles
    # if left_fit is not None:
    #     # Check if it's a line fit or circle fit
    #     if not (isinstance(left_fit, tuple) and len(left_fit) == 2 and left_fit[0] == 'line'):
    #         x0_l, y0_l, r_l = left_fit
    #         circle_theta = np.linspace(0, 2*np.pi, 300)
    #         circle_x_l = x0_l + r_l * np.cos(circle_theta)
    #         circle_y_l = y0_l + r_l * np.sin(circle_theta)
    #         ax.plot(circle_x_l, circle_y_l, 'b--', linewidth=2.5, 
    #                 label=f'Left Circle (r={r_l:.1f}, RMSE={left_errors["rmse_radial"]:.2f}, RelRMSE={left_errors["rel_rmse"]:.3f})', alpha=0.7)
    #         ax.plot(x0_l, y0_l, 'b+', markersize=12, markeredgewidth=2.5, label='Left Center')
        
    # if right_fit is not None:
    #     # Check if it's a line fit or circle fit
    #     if not (isinstance(right_fit, tuple) and len(right_fit) == 2 and right_fit[0] == 'line'):
    #         x0_r, y0_r, r_r = right_fit
    #         circle_theta = np.linspace(0, 2*np.pi, 300)
    #         circle_x_r = x0_r + r_r * np.cos(circle_theta)
    #         circle_y_r = y0_r + r_r * np.sin(circle_theta)
    #         ax.plot(circle_x_r, circle_y_r, 'r--', linewidth=2.5, 
    #                 label=f'Right Circle (r={r_r:.1f}, RMSE={right_errors["rmse_radial"]:.2f}, RelRMSE={right_errors["rel_rmse"]:.3f})', alpha=0.7)
    #         ax.plot(x0_r, y0_r, 'r+', markersize=12, markeredgewidth=2.5, label='Right Center')
    
    # Draw tangent lines at intersection points
    def draw_tangent_line(fit_result, intersection, side, color):
        if fit_result is None:
            return
        
        # Check if this is a line fit or circle fit
        if isinstance(fit_result, tuple) and len(fit_result) == 2 and fit_result[0] == 'line':
            # Line fit case
            _, line_params = fit_result
            slope, intercept, direction = line_params
            
            # Use the direction vector to draw the tangent (the line itself is the tangent)
            tangent_x = direction[0]
            tangent_y = direction[1]
            
            # Normalize tangent vector
            tangent_length_norm = np.sqrt(tangent_x**2 + tangent_y**2)
            if tangent_length_norm > 1e-10:
                tangent_x /= tangent_length_norm
                tangent_y /= tangent_length_norm
            
            # Create tangent line points with increased length
            line_length = 300  # Same as circle fit
            tangent_line_x = [intersection[0] - line_length * tangent_x, 
                              intersection[0] + line_length * tangent_x]
            tangent_line_y = [intersection[1] - line_length * tangent_y, 
                              intersection[1] + line_length * tangent_y]
            
            ax.plot(tangent_line_x, tangent_line_y, color=color, 
                    linewidth=3, label=f'{side.capitalize()} Tangent (Line Fit)', linestyle='-')
        else:
            # Circle fit case (original code)
            x0, y0, r = fit_result
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
            tangent_x /= tangent_length
            tangent_y /= tangent_length
            
            # Create tangent line points with increased length
            line_length = 300  # Increased from 150
            tangent_line_x = [intersection[0] - line_length * tangent_x, 
                              intersection[0] + line_length * tangent_x]
            tangent_line_y = [intersection[1] - line_length * tangent_y, 
                              intersection[1] + line_length * tangent_y]
            
            ax.plot(tangent_line_x, tangent_line_y, color=color, 
                    linewidth=3, label=f'{side.capitalize()} Tangent', linestyle='-')
    
    # Draw left and right tangents in cyan
    draw_tangent_line(left_fit, left_intersection, 'left', 'cyan')
    draw_tangent_line(right_fit, right_intersection, 'right', 'cyan')
    
    # Draw baseline (horizontal line through both intersections)
    baseline_x_min = left_intersection[0] - 50
    baseline_x_max = right_intersection[0] + 50
    ax.plot([baseline_x_min, baseline_x_max], [baseline_y, baseline_y], 
            'k-', linewidth=2.5, label='Baseline', alpha=0.8)
    
    # Add contact angle annotations
    ax.text(left_intersection[0] - 80, left_intersection[1] - 50, 
            f'θ_L = {left_angle:.1f}°', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    ax.text(right_intersection[0] + 40, right_intersection[1] - 50, 
            f'θ_R = {right_angle:.1f}°', fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
    
    # Add title with results
    ax.set_title(f'Circle Fit Contact Angle Calculation (ca.csv)\n' + 
                 f'Left: {left_angle:.2f}° (RelRMSE={left_errors["rel_rmse"]:.3f}) | ' +
                 f'Right: {right_angle:.2f}° (RelRMSE={right_errors["rel_rmse"]:.3f}) | Average: {avg_angle:.2f}°',
                 fontsize=15, fontweight='bold')
    
    # # Labels and formatting (COMMENTED OUT - axis labels and values)
    # ax.set_xlabel('X (pixels)', fontsize=12)
    # ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    # ax.grid(True, alpha=0.3)  # Grid removed
    ax.set_aspect('equal', adjustable='box')
    
    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Invert y-axis to match image coordinates (y increases downward)
    # ax.invert_yaxis()
    
    plt.tight_layout()
    plt.show()

    
    
    # Average contact angle
    avg_contact_angle = (left_angle + right_angle) / 2

    # Create the text content
    text_content = (
        f"Left Contact Angle: {left_angle:.2f} degrees\n"
        f"Right Contact Angle: {right_angle:.2f} degrees\n"
        f"Average Contact Angle: {avg_contact_angle:.2f} degrees\n"
    )

    # File path to save the data
    file_path = "Static_Contact_Angle.txt"

    # Write the data to a text file
    with open(file_path, "w") as file:
        file.write(text_content)
    return avg_contact_angle

# Function to display image and let user select baseline using a slider
def select_baseline(image):
    '''
    Provides an interactive slider for the user to select the baseline y-coordinate.

    Args:
        image: Input image to display for baseline selection.

    Returns:
        Selected baseline y-coordinate.
    '''
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
    # # Access the current figure's Tkinter window
    # canvas = plt.gcf().canvas
    # tk_window = canvas.manager.window
    # # Set the window size and position using Tkinter's geometry method
    # tk_window.geometry("960x540+0+0")
    plt.show()

    return int(baseline_slider.val)


# Full function to process the image, select baseline, and calculate contact angle
def process_image(image):
    '''
    Orchestrates the process of cropping, baseline selection, and contact angle calculation.

    Args:
        image: Input image of the droplet.
    '''
    if image is None:
        raise ValueError("Input image is None. Please provide a valid image.")
    global cropped_image
    # Get the cropped image from the user
    cropped_image = get_cropped_image(image)
    # Check if the user has skipped cropping
    if np.array_equal(cropped_image, image):
        # User chose not to crop, proceed with original image
        print("No cropping done. Using the original image.")
    get_threshes(cropped_image)
    # Select baseline using slider
    baseline_y = select_baseline(cropped_image)

    # Calculate contact angle
    avg_contact_angle = calculate_contact_angle(cropped_image, baseline_y)
    
    if avg_contact_angle is None:
        print("Contact angle calculation failed.")

def get_latest_file(directory):
    """
    Get the most recent file from the specified directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        str: Path to the latest file, or None if no files are found.
    """
    try:
        # Convert directory path to a Path object
        path = Path(directory)

        # List all files in the directory
        files = [f for f in path.iterdir() if f.is_file()]

        if not files:
            print(f"No files found in {directory}.")
            return None

        # Find the most recent file based on modification time
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        return str(latest_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_latest_image_path():
    # Replace with the directory where files are stored on your PC
    DIRECTORY = r"C:\Users\91982\OneDrive\Desktop\SURA\Retrieved_Data_Input"

    latest_file = get_latest_file(DIRECTORY)
    if latest_file:
        return latest_file
    else:
        print("No file could be retrieved.")

def sessile_drop():
    # image_retrieval_from_phone()
    # os.chdir(r"C:\Users\91982\OneDrive\Desktop\SURA")
    # image_path= str(load_latest_image_path())
    image_path= r"/Users/shyamkaushik/Desktop/Golden Days/Gonio/Inputs/Static Contact Angle Input Images/Asymmetric Droplets/tsd1.png"
    # image_path = r"C:\Users\91982\Downloads\no annotation 2.png" 
    image = cv2.imread(image_path)
    process_image(image)

sessile_drop()

