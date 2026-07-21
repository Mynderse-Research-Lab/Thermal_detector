#dashboard for thermal camera monitoring
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import csv
import os
import json
import threading
import time
from datetime import datetime
from flask import Flask, Response
import cv2
import numpy as np
import argparse
import io
import sys
import math
from pathlib import Path
#this code is intended to recieve live camera input from a thermal camera and, based on the pack input from the user
#and partition and monitor the battery pack temperature
MAX_RISE_C_PER_S = 2.0
MAX_TEMP_THRESHOLD = 100
CAMERA_INDEX=0



current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") #grab the date/time the file was made
csv_file_name = f"thermal_data_{current_timestamp}.csv"

script_dir = Path(__file__).parent #directory of current file
data_folder_path = script_dir / "raw_thermal_data" 
data_folder_path.mkdir(exist_ok=True)

frame_lock = threading.Lock()
latest_frame = None
latest_stats = {
    "max_temp": 0.0,
    "avg_temp": 0.0,
    "rise_rate": 0.0,
    "max_temp_warning": False,
    "thermal_runaway_warning": False,
    "roi_max": 0.0,
    "roi_avg": 0.0,
}

prev_maxtemp = None
prev_time = None
rise_rate = 0.0
too_fast = False
reset = False
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

TEST_ROI = (100, 100, 200, 150)

def detect_pack_region(img):
    #convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #detect edges
    edges = cv2.Canny(blurred, 5, 200)
    #find outlines/contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, edges
    #choose largest contour as likely battery pack
    largest = max(contours, key=cv2.contourArea)
    #ignore tiny detections
    if cv2.contourArea(largest) < 200:
        return None, edges
    #get bounding rectangle around pack
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h), edges

def frame_to_temp_array(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    temp_img = 20.0 + (gray.astype(np.float32) / 255.0) * 80.0
    return temp_img

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
    #roi,  edges = detect_pack_region(temp_img)
    return {
        "mean": float(roi.mean()),
        "max": float(roi.max()),
    }


def maxtemp_warning(maxtemp):
    if maxtemp > MAX_TEMP_THRESHOLD:
        return True
    else:
        return False
    #return maxtemp > MAX_TEMP_THRESHOLD


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

            #could audio alert reduce speed of data return? (Each time the alert plays, processing is delayed by 5 sec until audio finishes playing?)
            '''
            if too_fast and (not pygame.mixer.music.get_busy()): #if the rate of temp change is too fast and the audio alert is not currently playing
                pygame.mixer.music.play(loops=1) #play the audio alert for a single loop
            '''
            prev_maxtemp = maxtemp #update previous max temp to current max temp for next iteration
            prev_time = now #update previous time to current time for next iteration

    if prev_maxtemp is None: #if this is the first iteration and we don't have a previous max temp reading
        prev_maxtemp = maxtemp #set previous max temp to current max temp to start tracking
        prev_time = now #set previous time to current time to start tracking

    return too_fast, rise_rate #return the thermal runaway warning flag and the current rate of temperature change

def process_frame(frame):
    temp_img = frame_to_temp_array(frame)

    pack_result, edges = detect_pack_region(frame)
    if pack_result is None:
        x, y, w, h = TEST_ROI
    else:
        x, y, w, h = pack_result
    
    roi_result = roi_stats(temp_img, x, y, w, h)

    max_temp = roi_result["max"]
    avg_temp = roi_result["mean"]

    too_hot = maxtemp_warning(max_temp)
    too_fast, rise_rate = thermal_runaway_warning(max_temp)

    #(x, y, w, h), edges = detect_pack_region(temp_img)

    #roi_result = roi_stats(temp_img, x, y, w, h)

    stats = {
        "max_temp": max_temp,
        "avg_temp": avg_temp,
        "rise_rate": rise_rate,
        "max_temp_warning": too_hot,
        "thermal_runaway_warning": too_fast,
        "roi_max": roi_result["max"],
        "roi_avg": roi_result["mean"],
        "roi_bounds": (x, y, w, h),
    }

    return stats

def draw_overlay(frame, stats):
    #output = frame.copy()
    temp_img = frame_to_temp_array(frame)
    normalized = cv2.normalize(
        temp_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    jet_frame = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    #(x, y, w, h), edges = detect_pack_region(frame)
    x, y, w, h = stats["roi_bounds"]

    if stats["max_temp_warning"] or stats["thermal_runaway_warning"]:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)

    cv2.rectangle(jet_frame, (x, y), (x + w, y + h), color, 2)

    cv2.putText(jet_frame,f"ROI Max: {stats['roi_max']:.1f} C", (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(jet_frame, f"Max: {stats['max_temp']:.1f} C", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(jet_frame, f"Avg: {stats['avg_temp']:.1f} C", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.putText(jet_frame, f"Rise Rate: {stats['rise_rate']:.2f} C/s", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return jet_frame

def log_data(timestamp, maxTemp, avgTemp, roiMax, roiAvg): #log thermal data to a CSV file
    #current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    #date_str = current_timestamp.replace(" ", "_")
    csv_file_path = data_folder_path / csv_file_name
    with open(csv_file_path, "a", newline="") as file: #open a file called "thermal_data_{date_str}.csv" in append mode, where date_str is the current day and time the file was made
        writer = csv.writer(file) #create a CSV writer object to write data to the file
        writer.writerow([timestamp, maxTemp, avgTemp, roiMax, roiAvg]) #write a new row to the CSV file

def camera_loop():
    global latest_frame, latest_stats

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y','V'))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT,384)
    #cap = cv2.VideoCapture("/dev/video" + str(dev), cv2.CAP_V4L)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return
    
    display_min_temp = 15.0
    display_max_temp = 50.0
    initialization_frames = 0
    INITIALIZATION_FRAME_COUNT = 25

    while camera.isOpened():
        try:
            ret, frame = camera.read()
            print("Frame shape:", frame.shape, "dtype:", frame.dtype)

            if not ret:
                continue

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
            #elapse = time.monotonic() - start_time

            # Allow the camera to stabilize before locking the display range.
            initialization_frames += 1

            if (
                display_min_temp is None
                and initialization_frames >= INITIALIZATION_FRAME_COUNT
            ):
                # Percentiles prevent one noisy pixel from defining the entire scale.
                display_min_temp = float(np.percentile(temp_img, 2))
                display_max_temp = float(np.percentile(temp_img, 98))

                # Prevent a nearly uniform initial image from creating a tiny range.
                if display_max_temp - display_min_temp < 5.0:
                    display_max_temp = display_min_temp + 5.0

                print(
                    f"\nColormap locked at "
                    f"{display_min_temp:.1f} to {display_max_temp:.1f} C"
                )

            # Use a temporary range while waiting for the initial lock.
            if display_min_temp is None:
                frame_min = float(np.percentile(temp_img, 2))
                frame_max = float(np.percentile(temp_img, 98))

                if frame_max - frame_min < 5.0:
                    frame_max = frame_min + 5.0
            else:
                # These values remain fixed after initialization.
                frame_min = display_min_temp
                frame_max = display_max_temp

            # Clip temperatures to the fixed display range.
            clipped_temp = np.clip(
                temp_img,
                frame_min,
                frame_max
            )

            # Convert temperature values to the 0–255 colormap range.
            normalized_temp = (
                (clipped_temp - frame_min)
                / (frame_max - frame_min)
                * 255.0
            ).astype(np.uint8)

            # Apply JET to the actual temperature data.
            heatmap = cv2.applyColorMap(
                normalized_temp,
                cv2.COLORMAP_JET
            )

            heatmap = cv2.resize(
                heatmap,
                (newWidth, newHeight),
                interpolation=cv2.INTER_CUBIC
            )

            if rad > 0:
                heatmap = cv2.blur(heatmap, (rad, rad))

            cmapText = (
                f"Jet {frame_min:.1f}-{frame_max:.1f} C"
            )

            if not ret:
                print("WARNING: Failed to read camera frame.")
                time.sleep(0.1)
                continue
            '''
            # ROI definition (top-left x,y)
            roi_x, roi_y, roi_w, roi_h = 50, 50, 70, 30 # <------------------------------------------------------------------- Change the ROI

            # ROI stats
            stats = roi_stats(temp_img, roi_x, roi_y, roi_w, roi_h)

            # Draw ROI on heatmap (scaled)
            sx, sy = roi_x * scale, roi_y * scale
            sw, sh = roi_w * scale, roi_h * scale

            cv2.rectangle(heatmap, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.putText(heatmap, f"ROI max: {stats['max']:.1f} C", (sx, max(0, sy - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 255, 0), 1, cv2.LINE_AA)
                '''
            
            # ROI size in the original 256 × 192 temperature image
            roi_half_width = 20
            roi_half_height = 20

            # Keep the ROI inside temp_img
            x1 = max(0, mcol - roi_half_width)
            y1 = max(0, mrow - roi_half_height)
            x2 = min(temp_img.shape[1], mcol + roi_half_width)
            y2 = min(temp_img.shape[0], mrow + roi_half_height)

            # Calculate statistics using ORIGINAL, unscaled coordinates
            stats = roi_stats(
                temp_img,
                x1,
                y1,
                x2 - x1,
                y2 - y1
            )

            # Scale coordinates only when drawing on the resized heatmap
            top_left = (x1 * scale, y1 * scale)
            bottom_right = (x2 * scale, y2 * scale)

            cv2.rectangle(
                heatmap,
                top_left,
                bottom_right,
                (0, 255, 0),
                2
            )
            #= roi_stats(temp_img, roi_x - 50, roi_y - 50, 100, 100)
            
            # Crosshairs
            cv2.line(heatmap, (newWidth // 2, newHeight // 2 + 20), (newWidth // 2, newHeight // 2 - 20), (255, 255, 255), 2)
            cv2.line(heatmap, (newWidth // 2 + 20, newHeight // 2), (newWidth // 2 - 20, newHeight // 2), (255, 255, 255), 2)
            cv2.line(heatmap, (newWidth // 2, newHeight // 2 + 20), (newWidth // 2, newHeight // 2 - 20), (0, 0, 0), 1)
            cv2.line(heatmap, (newWidth // 2 + 20, newHeight // 2), (newWidth // 2 - 20, newHeight // 2), (0, 0, 0), 1)

            # Center temp text
            cx, cy = newWidth // 2, newHeight // 2
            cv2.putText(heatmap, f"{temp} C", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(heatmap, f"{temp} C", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # HUD
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
            #cv2.imshow("Thermal", heatmap)


            dashboard_stats = {
                "max_temp": maxtemp,
                "avg_temp": avgtemp,
                "rise_rate": rise_rate,
                "max_temp_warning": maxtemp_flag,
                "thermal_runaway_warning": too_fast,
                "roi_max": stats["max"],
                "roi_avg": stats["mean"],
            }

            with frame_lock:
                latest_frame = heatmap
                latest_stats = dashboard_stats

            log_data(datetime.now().isoformat(timespec="seconds"), dashboard_stats["max_temp"], dashboard_stats["avg_temp"], dashboard_stats["roi_max"], dashboard_stats["roi_avg"])

            time.sleep(0.05)
        except Exception as e:
            print("ERROR in camera_loop:", e)
            time.sleep(1)



def generate_video_stream():
    while True:
        with frame_lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

server = Flask(__name__)
app = Dash(__name__, server=server)

@server.route("/video_feed")

def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

app.layout = html.Div([
    html.H1("Thermal Camera Input & Measurements"),
    html.Div([html.Label("Pack type:", style={"fontWeight": "bold", "marginRight": "8px"}),
              dcc.Dropdown(['Jeep Wrangler PHEV', 'Ford Maverick HEV', 'Fiat 500e BEV'], 'Jeep Wrangler PHEV', id='pack-type-dropdown', closeOnSelect = True, searchable=False, multi=False), html.Div(id='dropdown-output')], style={"width":"400px", "marginBottom":"20px"}),
    html.H2("Live Thermal Camera Input"),
    html.Img(
        src="/video_feed",
        style={
            "width": "75%",
            "height": "auto",
            "maxHeight": "80vh",
            "objectFit": "contain",
            "display": "block",
            "border": "2px solid black"
        }),
    html.H2("Thermal Stats"),
    html.Div(id="stats-display"),
    dcc.Interval(
        id="stats-update-interval",
        interval=500,
        n_intervals=0)], 
    style={
    "width": "100%",
    "overflow": "hidden",
    "maxWidth": "none",
    "margin": "0",
    "padding": "20px",
    "boxSizing": "border-box"
})

@app.callback(
        Output('stats-display', 'children'),
        Input("stats-update-interval", "n_intervals"),
        Input('pack-type-dropdown', 'value')
)

def update_stats_display(n_intervals, pack_type):
    global too_fast, reset
    with frame_lock:
        stats = latest_stats.copy()

    if too_fast:
        status_text = f"DANGER / STOP"
    else:
        status_text = f"NORMAL"
    #danger = stats["max_temp_warning"] or stats["thermal_runaway_warning"]

    #status_text = "DANGER / STOP" if danger else "Normal"

    return html.Div([
        html.P(f"Selected Pack Type: {pack_type}"),
        html.P(f"System Status: {status_text}"),
        html.P(f"Max Temperature: {stats['max_temp']:.2f} °C"),
        html.P(f"Average Temperature: {stats['avg_temp']:.2f} °C"),
        html.P(f"Temperature Rise Rate: {stats['rise_rate']:.2f} °C/s"),
        html.P(f"Max Temp Warning: {stats['max_temp_warning']}"),
        html.P(f"Thermal Runaway Warning: {stats['thermal_runaway_warning']}"),
        html.P(f"Test ROI Max Temperature: {stats['roi_max']:.2f} °C"),
        html.P(f"Test ROI Average Temperature: {stats['roi_avg']:.2f} °C"),
    ])

if __name__ == '__main__':
    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()
    app.run(host="0.0.0.0", port=8050, debug=False)
