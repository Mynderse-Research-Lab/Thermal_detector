# Thermal Detector (OpenCV)

A lightweight thermal visualization + safety monitoring script that reads a **combined visible + thermal** frame from a V4L2 camera device (e.g., a thermal module that outputs YUYV + 16-bit thermal data in a single stream). The script:

- Displays a scaled **heatmap** window with crosshairs and an optional HUD
- Computes **center pixel**, **average**, **min**, **max** temperatures
- Tracks **thermal runaway risk** using max-temp rise rate (°C/s)
- Draws a configurable **ROI** box and reports ROI max/mean temperature
- Logs periodic temperature data to `thermal_data.csv`

---

## Features

### Temperature calculations
The thermal sensor provides a per-pixel 16-bit value (`raw_img`) that is converted to Celsius:

- Kelvin = `raw / 64.0`
- Celsius = `(raw / 64.0) - 273.15`

The script computes:
- **Center temperature**: pixel at `[96, 128]` (center of a 192x256 thermal image)
- **Max temperature & location**
- **Min temperature & location**
- **Average temperature** over the full thermal image
- **ROI max/mean** inside a user-defined rectangle

### Safety warnings
Two independent warnings are produced:

1. **Max Temperature Warning**
   - Trigger: `maxtemp > MAX_TEMP_THRESHOLD`
   - Default threshold: `MAX_TEMP_THRESHOLD = 100` °C

2. **Thermal Runaway Warning (Rise Rate)**
   - Trigger: max temperature rise rate exceeds a limit
   - Default limit: `MAX_RISE_C_PER_S = 2.0` °C/s
   - Rise rate is updated when `dt >= 0.50 s` to reduce noisy flicker.
   
## Development Setup (VS Code)
This project was developed and tested using **Visual Studio Code** for editing, debugging, and running the thermal detector script.

---
## Setup an venv
```bash
python3 -m venv venv
source venv/bin/activate
```
## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy

Install example:
```bash
pip install opencv-python numpy
pip install numpy
```
## Program Start
Default device is 0 witch is the built-in webcamera. The Thermal camera is normaly device 4
```bash
./thermal_detector.py --device 4
```
## Tip
To find the correct device uses this command below:
```bash
v4l2-ctl --list-devices
```

