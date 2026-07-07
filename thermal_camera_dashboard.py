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

TEST_ROI = (100, 100, 200, 150)

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

    max_temp = float(np.max(temp_img))
    avg_temp = float(np.mean(temp_img))

    too_hot = maxtemp_warning(max_temp)
    too_fast, rise_rate = thermal_runaway_warning(max_temp)

    x, y, w, h = TEST_ROI
    roi_result = roi_stats(temp_img, x, y, w, h)

    stats = {
        "max_temp": max_temp,
        "avg_temp": avg_temp,
        "rise_rate": rise_rate,
        "max_temp_warning": too_hot,
        "thermal_runaway_warning": too_fast,
        "roi_max": roi_result["max"],
        "roi_avg": roi_result["mean"],
    }

    return stats

def draw_overlay(frame, stats):
    output = frame.copy()

    x, y, w, h = TEST_ROI

    if stats["max_temp_warning"] or stats["thermal_runaway_warning"]:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)

    cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

    cv2.putText(
        output,
        f"ROI Max: {stats['roi_max']:.1f} C",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    cv2.putText(
        output,
        f"Max: {stats['max_temp']:.1f} C",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.putText(
        output,
        f"Avg: {stats['avg_temp']:.1f} C",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.putText(
        output,
        f"Rise Rate: {stats['rise_rate']:.2f} C/s",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    return output

def log_data(timestamp, maxTemp, avgTemp, roiMax, roiAvg): #log thermal data to a CSV file
    #current_timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    #date_str = current_timestamp.replace(" ", "_")
    csv_file_path = data_folder_path / csv_file_name
    with open(csv_file_path, "a", newline="") as file: #open a file called "thermal_data_{date_str}.csv" in append mode, where date_str is the current day and time the file was made
        writer = csv.writer(file) #create a CSV writer object to write data to the file
        writer.writerow([timestamp, maxTemp, avgTemp, roiMax, roiAvg]) #write a new row to the CSV file

def camera_loop():
    global latest_frame, latest_stats

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open camera.")
        return

    while True:
        ret, frame = camera.read()

        if not ret:
            print("WARNING: Failed to read camera frame.")
            time.sleep(0.1)
            continue

        stats = process_frame(frame)
        overlay_frame = draw_overlay(frame, stats)

        with frame_lock:
            latest_frame = overlay_frame
            latest_stats = stats

        log_data(stats)

        time.sleep(0.05)


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
            "width": "800px",
            "border": "2px solid black"
        }),
    html.H2("Thermal Stats"),
    html.Div(id="stats-display"),
    dcc.Interval(
        id="stats-update-interval",
        interval=500,
        n_intervals=0)
])

@app.callback(
        Output('stats-display', 'children'),
        Input("stats-update-interval", "n_intervals"),
        Input('pack-type-dropdown', 'value')
)

def update_stats_display(n_intervals, pack_type):
    with frame_lock:
        stats = latest_stats.copy()

    danger = stats["max_temp_warning"] or stats["thermal_runaway_warning"]

    status_text = "DANGER / STOP" if danger else "Normal"

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
