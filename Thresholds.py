import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import Slider
from ADB_extractor import image_retrieval_from_phone

# Global list to keep track of the contour plot objects
contour_lines = []

def log_thresholds(lower_threshold, upper_threshold):
    """Logs the final threshold values to a text file when the window is closed."""
    with open('thresholds_log.txt', 'w') as file:
        file.write(f"Final Lower Threshold: {lower_threshold}, Final Upper Threshold: {upper_threshold}")

def update(val, slider_lower, slider_upper, ax, fig, image, gray_image):
    """Update function to adjust the Canny thresholds and update contour display."""
    # Get the current values from the sliders
    lower_threshold = int(slider_lower.val)
    upper_threshold = int(slider_upper.val)

    # Apply Canny edge detection with the selected thresholds
    edges = cv2.Canny(gray_image, lower_threshold, upper_threshold)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove the previous contour lines from the plot
    for contour in contour_lines:
        contour.remove()
    contour_lines.clear()

    # Display the updated image with contours
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Canny Thresholds: {lower_threshold} - {upper_threshold}')
    
    # Draw the new contours on the image
    for contour in contours:
        contour_lines.append(ax.plot([point[0][0] for point in contour], 
                                     [point[0][1] for point in contour], color='g', linewidth=2)[0])

    # Redraw the image with contours
    fig.canvas.draw_idle()

    # Pause for a short while to reduce lag
    plt.pause(0.01)

    return lower_threshold, upper_threshold, contours

def on_close(event,slider_lower,slider_upper):
    """Callback function when the window is closed to log the final thresholds."""
    lower_threshold = int(slider_lower.val)
    upper_threshold = int(slider_upper.val)
    log_thresholds(lower_threshold, upper_threshold)
    print("Final thresholds logged.")

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

def get_threshes(image):

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    # Display the image
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title('Adjust the Canny Thresholds')

    # Add sliders for adjusting the lower and upper threshold values
    ax_slider_lower = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_lower = Slider(ax_slider_lower, 'Lower Threshold', 0, 255, valinit=100, valstep=1)

    ax_slider_upper = plt.axes([0.1, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider_upper = Slider(ax_slider_upper, 'Upper Threshold', 0, 255, valinit=200, valstep=1)

    # Attach the update function to the sliders
    slider_lower.on_changed(lambda val: update(val, slider_lower, slider_upper, ax, fig, image, gray_image))
    slider_upper.on_changed(lambda val: update(val, slider_lower, slider_upper, ax, fig, image, gray_image))


    # Connect the close event to log the final thresholds
    fig.canvas.mpl_connect('close_event', lambda event: on_close(event, slider_lower, slider_upper))
    # # Access the current figure's Tkinter window
    # canvas = plt.gcf().canvas
    # tk_window = canvas.manager.window
    # # Set the window size and position using Tkinter's geometry method
    # tk_window.geometry("960x540+0+0")  # Position at (0, 0) with size 960x540
    # # Show the plot
    plt.show()

def Threshold():
    image_retrieval_from_phone()
    os.chdir(r"C:\Users\91982\OneDrive\Desktop\SURA")
    image_path= str(load_latest_image_path())
    image_path = r"C:\Users\91982\Downloads\obtuse.jpg"
    image = cv2.imread(image_path)
    get_threshes(image)
