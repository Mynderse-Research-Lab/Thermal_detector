import cv2
import numpy as np
import json
from pathlib import Path
from datetime import datetime

RGB_CAMERA_INDEX = 0
THERMAL_CAMERA_INDEX = 1  #change if necessary
JSON_FILE = (Path(__file__).resolve.parent / "dual-cam_transform-coord.json")

rgb_calibration_points = []
thermal_calibration_points = []

rgb_crosshair = None
thermal_crosshair = None

def draw_crosshair(frame, point, label):
    if point is None: #if there is no 
        return frame

    output = frame.copy() #make a copy of the camera output frame
    x, y = point

    color = (0, 255, 0)
    crosshair_size = 15

    cv2.line( #draw the crosshair horizontal line on the camera output frame copy
        output,
        (x - crosshair_size, y),
        (x + crosshair_size, y),
        color,
        2
    )

    cv2.line( #draw the crosshair vertical line on the camera output frame copy
        output,
        (x, y - crosshair_size),
        (x, y + crosshair_size),
        color,
        2
    )

    cv2.circle( #draw the center circle on the crosshair on the camera output frame copy
        output,
        (x, y),
        4,
        color,
        2
    )

    cv2.putText( #label pixel location on frame copy
        output,
        f"{label}: ({x}, {y})",
        (x + 12, max(y - 12, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
    )

    return output


def rgb_mouse_callback(event, x, y, flags, parameter): #mouse interrupt routine on rgb image
    global rgb_crosshair

    if event == cv2.EVENT_LBUTTONDOWN: #if a left click triggers an interrupt
        rgb_crosshair = (x, y) #record the pixel coordinate location x,y
        rgb_calibration_points.append([float(x), float(y)])
        print(f"RGB point:     [{x}, {y}]") #print location


def thermal_mouse_callback(event, x, y, flags, parameter): #mouse interrupt routine on thermal image
    global thermal_crosshair

    if event == cv2.EVENT_LBUTTONDOWN: #if a left click triggers an interrupt
        thermal_crosshair = (x, y) #record the pixel coordinate location x,y
        thermal_calibration_points.append([float(x),float(y)])
        print(f"Thermal point: [{x}, {y}]") #print location

def save_calibration(rgb_frame_shape, thermal_frame_shape):
    if len(rgb_calibration_points) < 4:
        print("Calibration requires at least four RGB points")
        return False
    if len(thermal_calibration_points) < 4:
        print("Calibration requires at least four Thermal points")
        return False
    if(len(rgb_calibration_points) != len(thermal_calibration_points)):
        print("The number of RGB and thermal points must be equal.")
        print(f"RGB points: {len(rgb_calibration_points)}")
        print(f"Thermal points:{len(thermal_calibration_points)}")
        return False
    rgb_points_array = np.asarray(rgb_calibration_points, dtype=np.float32)
    thermal_points_array = np.asarray(thermal_calibration_points, dtype=np.float32)
    thermal_to_rgb_matrix, inlier_mask = (
        cv2.findHomography(thermal_points_array, rgb_points_array, method=cv2.RANSAC, ransacReprojThreshold=3.0))
    if thermal_to_rgb_matrix is None:
        print("Could not calculate the homography")
        return False
    rgb_height, rgb_width = rgb_frame_shape[:2]
    thermal_height, thermal_width = thermal_frame_shape[:2]
    calibration_data = {
        "version": 1,
        "created": datetime.now().isoformat(timespec="seconds"),
        "coord_sys": {
            "rgb":{
                "width": int(rgb_width),
                "height": int(rgb_height)},
            "thermal":{
                "width": int(thermal_width),
                "height": int(thermal_height)}},
        "rgb_points": (rgb_points_array.tolist()),
        "thermal_points": (thermal_points_array.tolist()),
        "thermal_to_rgb_matrix": (thermal_to_rgb_matrix.tolist()),
        "ransac_inliers": (
            inlier_mask.reshape(-1).astype(int).tolist()
            if inlier_mask is not None
            else None
        )
    }
    temporary_file = JSON_FILE.with_suffix(".json.tmp")
    temporary_file.write_text(json.dumps(calibration_data, indent=4), encoding="utf-8")
    temporary_file.replace(JSON_FILE)
    print(f"Saved calibration to: {JSON_FILE}")
    print(f"thermal to rgb matrix: {thermal_to_rgb_matrix}")
    return True


rgb_camera = cv2.VideoCapture( #grab camera output from rgb
    RGB_CAMERA_INDEX,
    cv2.CAP_V4L2
)

thermal_camera = cv2.VideoCapture( #grab camera output from thermal
    THERMAL_CAMERA_INDEX,
    cv2.CAP_V4L2
)

#webcam settings
rgb_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
rgb_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#thermal cam raw-frame settings
thermal_camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
thermal_camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc("Y", "U", "Y", "V")
)
thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

#error checking for cam output
if not rgb_camera.isOpened():
    raise RuntimeError("Could not open RGB camera")

if not thermal_camera.isOpened():
    raise RuntimeError("Could not open thermal camera")

cv2.namedWindow("RGB Calibration", cv2.WINDOW_NORMAL)
cv2.namedWindow("Thermal Calibration", cv2.WINDOW_NORMAL)

cv2.setMouseCallback(
    "RGB Calibration",
    rgb_mouse_callback
)

cv2.setMouseCallback(
    "Thermal Calibration",
    thermal_mouse_callback
)

while True:
    rgb_success, rgb_frame = rgb_camera.read()
    thermal_success, combined_frame = thermal_camera.read()

    if not rgb_success or rgb_frame is None:
        print("Could not read RGB frame")
        break

    if not thermal_success or combined_frame is None:
        print("Could not read thermal frame")
        break

    # TC001 provides a 384-row frame containing two 192-row sections.
    visible_data, thermal_data = np.array_split(
        combined_frame,
        2
    )

    # Decode the TC001 16-bit temperature data.
    high_byte = thermal_data[..., 1].astype(np.uint16)
    low_byte = thermal_data[..., 0].astype(np.uint16)

    raw_temperature = (
        high_byte << 8
    ) | low_byte

    temperature_c = (
        raw_temperature.astype(np.float32) / 64.0
    ) - 273.15

    display_min = 10.0
    display_max = 50.0

    clipped = np.clip(
        temperature_c,
        display_min,
        display_max
    )

    normalized = (
        (clipped - display_min)
        / (display_max - display_min)
        * 255
    ).astype(np.uint8)

    thermal_frame = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_JET
    )

    rgb_display = draw_crosshair(
        rgb_frame,
        rgb_crosshair,
        "RGB"
    )

    thermal_display = draw_crosshair(
        thermal_frame,
        thermal_crosshair,
        "Thermal"
    )

    cv2.imshow(
        "RGB Calibration",
        rgb_display
    )

    cv2.imshow(
        "Thermal Calibration",
        thermal_display
    )

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"): save_calibration(rgb_frame.shape, thermal_frame.shape)

    if key == ord("q"): break

    if key == ord("u"):
        if rgb_calibration_points:
            removed_rgb = (rgb_calibration_points.pop())
            print(f"Removed point: {removed_rgb}")
        if thermal_calibration_points:
            removed_thermal=(thermal_calibration_points.pop())
            print(f"Removed thermal point: {removed_thermal}")

    # Press C to clear both crosshairs.
    if key == ord("c"):
        rgb_crosshair = None
        thermal_crosshair = None

rgb_camera.release()
thermal_camera.release()
cv2.destroyAllWindows()