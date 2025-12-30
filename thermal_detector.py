#!/usr/bin/env python3
print('Thermal Detector')
print('Key Bindings:')
print('')
print('q: quit')
print('h : Toggle HUD')
print('c : Reset Thermal Runaway Alert')
print('')

import cv2
import numpy as np
import argparse
import time
import io
import sys

#We need to know if we are running on the Pi
def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False

def print_status(maxtemp, too_fast, rise_rate, avgtemp, maxtemp_flag):
    sys.stdout.write(
        f"\rMax Temp Warning: {maxtemp:.2f} C | Flag: {maxtemp_flag} | limit: {MAX_TEMP_THRESHOLD}\n"
        f"Ave Temp: {avgtemp:.2f} C\n"
        f"Thermal Runaway Warning: Rate: {rise_rate:.2f} [C/s] | Flag: {too_fast} | Limit: {MAX_RISE_C_PER_S:.2f} [C/s]\n"
        "\033[F\033[F\033[F"   # move cursor up 2 lines
    )
    sys.stdout.flush()

isPi = is_raspberrypi()


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
args = parser.parse_args()
	
if args.device:
	dev = args.device
else:
	dev = 0
	
#init video
cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)

if isPi == True:
	cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
	cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
	

#256x192 General settings
width = 256 #Sensor width
height = 192 #sensor height
scale = 3 #scale multiplier
newWidth = width*scale 
newHeight = height*scale
alpha = 1.0 # Contrast control (1.0-3.0)
colormap = 0
font=cv2.FONT_HERSHEY_SIMPLEX
dispFullscreen = False
cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Thermal', newWidth,newHeight)
rad = 0 #blur radius
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

def maxtemp_warning(maxtemp):
    if maxtemp > MAX_TEMP_THRESHOLD:
        return True
    return False

def thermal_runaway_warning(maxtemp):
    global prev_maxtemp, prev_time, rise_rate, too_fast, reset

    now = time.time()

    if prev_maxtemp is not None:
        dt = now - prev_time

        
        if dt >= 0.50:
            rise_rate = (maxtemp - prev_maxtemp) / dt
            
            if (not too_fast) or reset:
                reset = False
                too_fast = (rise_rate > MAX_RISE_C_PER_S)

            prev_maxtemp = maxtemp
            prev_time = now

    # First call initialization
    if prev_maxtemp is None:
        prev_maxtemp = maxtemp
        prev_time = now

    return too_fast, rise_rate



def rec():
	now = time.strftime("%Y%m%d--%H%M%S")
	#do NOT use mp4 here, it is flakey!
	videoOut = cv2.VideoWriter(now+'output.avi', cv2.VideoWriter_fourcc(*'XVID'),25, (newWidth,newHeight))
	return(videoOut)
 
while(cap.isOpened()):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
     
		imdata,thdata = np.array_split(frame, 2)

		hi_img = thdata[..., 1].astype(np.uint16)   # high byte
		lo_img = thdata[..., 0].astype(np.uint16)   # low byte
		raw_img = (hi_img << 8) | lo_img            # 16-bit raw per pixel

		# Center pixel temperature
		rawtemp = raw_img[96, 128]
		temp = (rawtemp / 64.0) - 273.15
		temp = round(float(temp), 2)

		# Max temperature + location
		max_idx = np.unravel_index(raw_img.argmax(), raw_img.shape)
		mcol, mrow = int(max_idx[0]), int(max_idx[1])   # row, col in image coords
		maxtemp_raw = raw_img[mcol, mrow]
		maxtemp = (maxtemp_raw / 64.0) - 273.15
		maxtemp = round(float(maxtemp), 2)
  
		# Min temperature + location
		min_idx = np.unravel_index(raw_img.argmin(), raw_img.shape)
		lcol, lrow = int(min_idx[0]), int(min_idx[1])
		mintemp_raw = raw_img[lcol, lrow]
		mintemp = (mintemp_raw / 64.0) - 273.15
		mintemp = round(float(mintemp), 2)

		# Average temperature
		avgtemp_raw = raw_img.astype(np.float32).mean()
		avgtemp = (avgtemp_raw / 64.0) - 273.15
		avgtemp = round(float(avgtemp), 2)
	

		maxtemp_flag = maxtemp_warning(maxtemp)
		too_fast, rise_rate = thermal_runaway_warning(maxtemp)
		print_status(maxtemp, too_fast, rise_rate, avgtemp, maxtemp_flag)


		# Convert the real image to RGB
		bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)
		#Contrast
		bgr = cv2.convertScaleAbs(bgr, alpha=alpha)#Contrast
		#bicubic interpolate, upscale and blur
		bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)#Scale up!
		if rad>0:
			bgr = cv2.blur(bgr,(rad,rad))

		#apply colormap
		if colormap == 0:
			heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
			cmapText = 'Jet'
		

		#print(heatmap.shape)

		# draw crosshairs
		cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
		(int(newWidth/2),int(newHeight/2)-20),(255,255,255),2) #vline
		cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
		(int(newWidth/2)-20,int(newHeight/2)),(255,255,255),2) #hline

		cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
		(int(newWidth/2),int(newHeight/2)-20),(0,0,0),1) #vline
		cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
		(int(newWidth/2)-20,int(newHeight/2)),(0,0,0),1) #hline
		#show temp
		cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
		cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
		cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
		cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

		if hud==True:
			# display black box for our data
			cv2.rectangle(heatmap, (0, 0),(160, 120), (0,0,0), -1)
			# put text in the box
			cv2.putText(heatmap,'Avg Temp: '+str(avgtemp)+' C', (10, 14),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Label Threshold: '+str(threshold)+' C', (10, 28),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Colormap: '+cmapText, (10, 42),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Blur: '+str(rad)+' ', (10, 56),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Scaling: '+str(scale)+' ', (10, 70),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

			cv2.putText(heatmap,'Contrast: '+str(alpha)+' ', (10, 84),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0, 255, 255), 1, cv2.LINE_AA)

		
		#display floating max temp
		if maxtemp > avgtemp+threshold:
			cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,0), 2)
			cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,255), -1)
			cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
			cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

		#display floating min temp
		if mintemp < avgtemp-threshold:
			cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (0,0,0), 2)
			cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (255,0,0), -1)
			cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
			cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
			cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

		#display image
		cv2.imshow('Thermal',heatmap)
		
		keyPress = cv2.waitKey(1)

		if keyPress == ord('h'):
			if hud==True:
				hud=False
			elif hud==False:
				hud=True
    
		if keyPress == ord('c'):
			reset = True

		if keyPress == ord('q'):
			cap.release()
			cv2.destroyAllWindows()
   			
