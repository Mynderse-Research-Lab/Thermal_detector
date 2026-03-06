#!/usr/bin/env python3

print("Thermal Detector")
print("Key Bindings:\n")
print("q: quit")
print("h: Toggle HUD")
print("c: Reset Thermal Runaway Alert")
print("")

import cv2
import numpy as np
import argparse
import time
import io
import sys
import math
import csv
from pathlib import Path
current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") #grab the date/time the file was made
csv_file_name = f"thermal_data_{current_timestamp}.csv"

script_dir = Path(__file__).parent #directory of current file
data_folder_path = script_dir / "raw_thermal_data" 
data_folder_path.mkdir(exist_ok=True)

# We need to know if we are running on the Pi
def is_raspberrypi(): #function definition: check if the raspberry pi is connected and able to be successfully opened.
    try: 
        with io.open("/sys/firmware/devicetree/base/model", "r") as m: #open i/o device location and read the hardware model information
            if "raspberry pi" in m.read().lower(): #if the string "raspberry pi" is located anywhere within the hardware model information file
                #print("Raspberry Pi detected") #print that the raspberry pi is connected
                return True #return true to indicate that we are connected to the raspberry pi
    except Exception: #in the case of an error being thrown
        pass #ignore error
    #print("No Raspberry Pi detected") #print that the raspberry pi is not connected
    return False #return false


def print_status(maxtemp, too_fast, rise_rate, avgtemp, maxtemp_flag): #function definitition: print the status of the thermal detector to the console
    sys.stdout.write( #write the following information to the console
        f"\rMax Temp Warning: {maxtemp:.2f} C | Flag: {maxtemp_flag} | limit: {MAX_TEMP_THRESHOLD}\n"
        f"Ave Temp: {avgtemp:.2f} C\n"
        f"Thermal Runaway Warning: Rate: {rise_rate:.2f} [C/s] | Flag: {too_fast} | "
        f"Limit: {MAX_RISE_C_PER_S:.2f} [C/s]\n"
        "\033[F\033[F\033[F" #move cursor up 3 lines to overwrite Avg temp, thermal runaway warning, and limit status
    )
    sys.stdout.flush() #flush the output buffer to print status immediately (no processing delay)


def roi_stats(temp_img, x, y, w, h): #calculate region of interest stats
    roi = temp_img[y:y+h, x:x+w]
    return {
        "mean": float(roi.mean()),
        "max": float(roi.max()),
    }


def maxtemp_warning(maxtemp):
    return maxtemp > MAX_TEMP_THRESHOLD


def thermal_runaway_warning(maxtemp):
    global prev_maxtemp, prev_time, rise_rate, too_fast, reset #set global variables to track temperature info

    now = time.time() #set current time from machine clock

    if prev_maxtemp is not None: #if we have a previous max temp reading (this isn't the first iteration)
        dt = now - prev_time #difference in time = current time - previous time

        if dt >= 0.50: #if the last reading was taken at least 0.5 seconds ago
            rise_rate = (maxtemp - prev_maxtemp) / dt #calculate the rate of temperature change = (current max temp - previous max temp) / time difference

            if (not too_fast) or reset: #if the rate of temperature change is not too fast, or the reset flag is set
                reset = False #unset the reset flag
                too_fast = (rise_rate > MAX_RISE_C_PER_S) #test if the rate of temp change is too fast by comparing to max rate threshold and set flag

            prev_maxtemp = maxtemp #update previous max temp to current max temp for next iteration
            prev_time = now #update previous time to current time for next iteration

    if prev_maxtemp is None: #if this is the first iteration and we don't have a previous max temp reading
        prev_maxtemp = maxtemp #set previous max temp to current max temp to start tracking
        prev_time = now #set previous time to current time to start tracking

    return too_fast, rise_rate #return the thermal runaway warning flag and the current rate of temperature change


def log_data(timestamp, maxTemp, avgTemp, roiMax, roiAvg): #log thermal data to a CSV file
    #current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    #date_str = current_timestamp.replace(" ", "_")
    csv_file_path = data_folder_path / csv_file_name
    with open(csv_file_path, "a", newline="") as file: #open a file called "thermal_data_{date_str}.csv" in append mode, where date_str is the current day and time the file was made
        writer = csv.writer(file) #create a CSV writer object to write data to the file
        writer.writerow([timestamp, maxTemp, avgTemp, roiMax, roiAvg]) #write a new row to the CSV file



# ----------------- main -----------------
isPi = is_raspberrypi() #test connection to raspberry pi

parser = argparse.ArgumentParser() #create an argument parser object to handle command line arguments
parser.add_argument( #add an argument for the video device number, with a default of 0 and a help message
    "--device",
    type=int,
    default=0,
    help="Video Device number e.g. 0, use v4l2-ctl --list-devices",
)
args = parser.parse_args() #parse the command line arguments and store them in the args variable

dev = args.device if args.device is not None else 0 #if the device argument is provided, use that device or default to 0

# init video
cap = cv2.VideoCapture("/dev/video" + str(dev), cv2.CAP_V4L) #open the video capture device at the specified path

if isPi: #if we are running on the raspberry pi, 
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0) #disable automatic RGB conversion to preserve the raw thermal data in the video feed
else: #if we are not running on the raspberry pi
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0) #disable automatic RGB conversion 

# 256x192 General settings
width = 256
height = 192
scale = 3
newWidth = width * scale
newHeight = height * scale

alpha = 1.0
colormap = 0
rad = 0
hud = True
threshold = 2

# thermal runaway tracking
prev_maxtemp = None
prev_time = None
rise_rate = 0.0
reset = False
too_fast = False

MAX_RISE_C_PER_S = 2.0
MAX_TEMP_THRESHOLD = 100

start_time = time.monotonic()
log_time = 0

cv2.namedWindow("Thermal", cv2.WINDOW_GUI_NORMAL) #create a named window for displaying the thermal feed
cv2.resizeWindow("Thermal", newWidth, newHeight) #allow resizing of the window


while cap.isOpened(): #while the video capture device is successfully opened and ready to read frames
    ret, frame = cap.read() #read a frame from the device
    if not ret: #if the frame was not read
        continue #stay in while loop

    # Split combined frame into visible + thermal parts
    imdata, thdata = np.array_split(frame, 2)

    # Decode 16-bit thermal raw
    hi_img = thdata[..., 1].astype(np.uint16)
    lo_img = thdata[..., 0].astype(np.uint16)
    raw_img = (hi_img << 8) | lo_img

    # Center pixel temp
    rawtemp = raw_img[96, 128]
    temp = (rawtemp / 64.0) - 273.15
    temp = round(float(temp), 2)

    # Full temp image (°C) per pixel
    temp_img = (raw_img.astype(np.float32) / 64.0) - 273.15

    # Max temp + location
    max_idx = np.unravel_index(raw_img.argmax(), raw_img.shape)
    mrow, mcol = int(max_idx[0]), int(max_idx[1])  # row, col
    maxtemp_raw = raw_img[mrow, mcol]
    maxtemp = (maxtemp_raw / 64.0) - 273.15
    maxtemp = round(float(maxtemp), 2)

    # Min temp + location
    min_idx = np.unravel_index(raw_img.argmin(), raw_img.shape)
    lrow, lcol = int(min_idx[0]), int(min_idx[1])
    mintemp_raw = raw_img[lrow, lcol]
    mintemp = (mintemp_raw / 64.0) - 273.15
    mintemp = round(float(mintemp), 2)

    # Avg temp
    avgtemp = float(temp_img.mean())
    avgtemp = round(avgtemp, 2)

    # Warnings + status
    maxtemp_flag = maxtemp_warning(maxtemp)
    too_fast, rise_rate = thermal_runaway_warning(maxtemp)
    print_status(maxtemp, too_fast, rise_rate, avgtemp, maxtemp_flag)
    
    # Calculate elapsed time
    elapse = time.monotonic() - start_time

    # Convert the real image to RGB for display
    bgr = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
    bgr = cv2.convertScaleAbs(bgr, alpha=alpha)
    bgr = cv2.resize(bgr, (newWidth, newHeight), interpolation=cv2.INTER_CUBIC)

    if rad > 0:
        bgr = cv2.blur(bgr, (rad, rad))

    # Apply colormap
    cmapText = "Jet"
    if colormap == 0:
        heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)

    # ROI definition (top-left x,y)
    roi_x, roi_y, roi_w, roi_h = 50, 50, 70, 30 # <------------------------------------------------------------------- Change the ROI

    # ROI stats
    stats = roi_stats(temp_img, roi_x, roi_y, roi_w, roi_h)

    # Draw ROI on heatmap (scaled)
    sx, sy = roi_x * scale, roi_y * scale
    sw, sh = roi_w * scale, roi_h * scale

    cv2.rectangle(heatmap, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
    cv2.putText(
        heatmap,
        f"ROI max: {stats['max']:.1f} C",
        (sx, max(0, sy - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    
    # Periodic logging
    if elapse >= 15.0:
        log_time = elapse + log_time
        log_data(
            math.floor(log_time),
            maxtemp,
            avgtemp,
            stats["max"],
            stats["mean"]
        )
        start_time = time.monotonic()

    # Crosshairs
    cv2.line(heatmap, (newWidth // 2, newHeight // 2 + 20), (newWidth // 2, newHeight // 2 - 20), (255, 255, 255), 2)
    cv2.line(heatmap, (newWidth // 2 + 20, newHeight // 2), (newWidth // 2 - 20, newHeight // 2), (255, 255, 255), 2)
    cv2.line(heatmap, (newWidth // 2, newHeight // 2 + 20), (newWidth // 2, newHeight // 2 - 20), (0, 0, 0), 1)
    cv2.line(heatmap, (newWidth // 2 + 20, newHeight // 2), (newWidth // 2 - 20, newHeight // 2), (0, 0, 0), 1)

    # Center temp text
    cx, cy = newWidth // 2, newHeight // 2
    cv2.putText(heatmap, f"{temp} C", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(heatmap, f"{temp} C", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # HUD #heads up display
    if hud: 
        cv2.rectangle(heatmap, (0, 0), (200, 120), (0, 0, 0), -1)
        cv2.putText(heatmap, f"Avg Temp: {avgtemp} C", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(heatmap, f"ROI Max: {stats['max']:.1f} C", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(heatmap, f"Colormap: {cmapText}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(heatmap, f"Scaling: {scale}", (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(heatmap, f"Contrast: {alpha}", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # Floating max temp marker
    if maxtemp > avgtemp + threshold:
        cv2.circle(heatmap, (mcol * scale, mrow * scale), 5, (0, 0, 0), 2)
        cv2.circle(heatmap, (mcol * scale, mrow * scale), 5, (0, 0, 255), -1)
        cv2.putText(heatmap, f"{maxtemp} C", (mcol * scale + 10, mrow * scale + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap, f"{maxtemp} C", (mcol * scale + 10, mrow * scale + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # Floating min temp marker
    if mintemp < avgtemp - threshold:
        cv2.circle(heatmap, (lcol * scale, lrow * scale), 5, (0, 0, 0), 2)
        cv2.circle(heatmap, (lcol * scale, lrow * scale), 5, (255, 0, 0), -1)
        cv2.putText(heatmap, f"{mintemp} C", (lcol * scale + 10, lrow * scale + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(heatmap, f"{mintemp} C", (lcol * scale + 10, lrow * scale + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    # Display image
    cv2.imshow("Thermal", heatmap)

    keyPress = cv2.waitKey(1) & 0xFF #wait for a key press and store the key code in the keyPress variable

    if keyPress == ord("h"): #if the "h" key is pressed, toggle the heads up display (HUD) on and off 
        hud = not hud

    if keyPress == ord("c"): # if the "c" key is pressed, set the reset flag to true
        reset = True

    if keyPress == ord("q"): #if the "q" key is pressed, exit the while loop and end the program
        break

cap.release() #release the video capture resources
cv2.destroyAllWindows() #close all OpenCV windows
