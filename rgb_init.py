import numpy as np
import cv2

def get_adjusted_thermal_to_rgb_matrix(use_RGB_flag, overlay_alignment_lock, overlay_x_offset, overlay_y_offset, base_thermal_to_rgb_matrix):
    if not use_RGB_flag: return None
    with overlay_alignment_lock:
        x_offset = float(overlay_x_offset)
        y_offset = float(overlay_y_offset)
    translation_matrix = np.array([[1.0, 0.0, x_offset],[0.0,1.0,y_offset],[0.0,0.0,1.0]], dtype=np.float64)
    adjusted_matrix = (translation_matrix @ base_thermal_to_rgb_matrix) #@ => matrix multiplication
    return adjusted_matrix

def overlay_thermal_on_rgb(rgb_frame, thermal_frame, thermal_to_rgb_matrix, alpha):
    rgb_height, rgb_width = rgb_frame.shape[:2]
    warped_thermal = cv2.warpPerspective(thermal_frame, thermal_to_rgb_matrix, (rgb_width, rgb_height), flags=cv2.INTER_LINEAR)
    thermal_mask = np.full(thermal_frame.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(thermal_mask, thermal_to_rgb_matrix, (rgb_width, rgb_height), flags=cv2.INTER_NEAREST)
    valid_pixels = warped_mask > 0
    combined_frame = rgb_frame.copy()
    blended = cv2.addWeighted(rgb_frame, 1.0-alpha, warped_thermal, alpha, 0)
    combined_frame[valid_pixels] = blended[valid_pixels]
    return combined_frame, warped_thermal

def map_thermal_points_to_rgb(thermal_points, transform_matrix, use_RGB_flag, overlay_alignment_lock, overlay_x_offset, overlay_y_offset, base_thermal_to_rgb_matrix):
    if transform_matrix is None:
        transform_matrix = (get_adjusted_thermal_to_rgb_matrix(use_RGB_flag, overlay_alignment_lock, overlay_x_offset, overlay_y_offset, base_thermal_to_rgb_matrix))
    points = np.asarray(thermal_points, dtype=np.float32).reshape(-1,1,2)
    rgb_points = cv2.perspectiveTransform(points, transform_matrix)
    return rgb_points.reshape(-1,2)