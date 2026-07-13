# Python program to illustrate
# template matching
import cv2
import numpy as np
import sys
from pathlib import Path

script_dir = Path(__file__).parent #directory of current file
temp_filename = 'jeep-1_template.png' #works for jeep-1.png, jeep-3.jpeg, 
temp_dir = script_dir / temp_filename
img_filename = 'jeep-3.jpeg'
img_dir = script_dir / img_filename

def template_matching(): #identify pack by template region - requires EXACT match for accuracy
    img_rgb = cv2.imread(img_dir) #load image
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) #convert it to grayscale
    template = cv2.imread(temp_dir, 0) #load the template image

    w, h = template.shape[::-1] #get width and height of template
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED) #perform match operations
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #get vals and locations of matches
    print(f"The highest match score found was: {max_val:.4f}")

    top_left = max_loc  #(x, y) of the best match
    bottom_right = (top_left[0] + w, top_left[1] + h) 

    threshold = 0.1 #certainty threshold for match

    loc = np.where(res >= threshold) #coordinates of matched area
    cv2.rectangle(img_rgb, top_left, bottom_right, (0, 255, 255), 20) #draw a rectangle around the matched region
    
    cv2.namedWindow('Detected', cv2.WINDOW_NORMAL) #display final image w/ resizable window
    cv2.imshow('Detected', img_rgb)

    cv2.waitKey(0) #wait indefinitely for keyboard input

    cv2.destroyAllWindows() #cleanup windows

def feature_matching(): #identify pack by matching prominent features
    img = cv2.imread(str(img_dir), cv2.IMREAD_GRAYSCALE) #load image and convert to grayscale
    template = cv2.imread(str(temp_dir), cv2.IMREAD_GRAYSCALE) #load the template image
    
    if img is None or template is None: #error catching
        sys.exit("Error: Could not load images. Check your filenames and paths.")

    orb = cv2.ORB_create(nfeatures=2000) #initialize the ORB (oriented FAST and rotated BRIEF) detector 
    
    #find keypoints and calculate descriptors for both images
    kp_template, des_template = orb.detectAndCompute(template, None) 
    kp_img, des_img = orb.detectAndCompute(img, None)

    #create a Brute-Force Matcher to compare descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_template, des_img)

    #sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    #filter for best matches
    good_matches = matches[:50] #50 best matches

    #calculate homography
    src_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    #draw bounding box around likely object using best matches
    if M is not None:
        h_temp, w_temp = template.shape
        pts_template = np.float32([[0, 0], [0, h_temp - 1], [w_temp - 1, h_temp - 1], [w_temp - 1, 0]]).reshape(-1, 1, 2)
        pts_scene = cv2.perspectiveTransform(pts_template, M)
        img_color = cv2.imread(str(img_dir))
        box_points = np.int32(pts_scene).reshape((-1, 1, 2))
        cv2.polylines(img_color, [box_points], isClosed=True, color=(0, 255, 0), thickness=10)
    else:
        print("Could not find enough matching features to calculate a bounding box.")
        img_color = cv2.imread(str(img_dir)) #fallback to show empty scene

    #display
    cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Object', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

feature_matching()