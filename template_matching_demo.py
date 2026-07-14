# Python program to illustrate
# template matching
import cv2
import numpy as np
import sys
from pathlib import Path

script_dir = Path(__file__).parent #directory of current file
temp_filename = 'jeep-1_template.png' #works for jeep-1.png, jeep-3.jpeg, 
temp_dir = script_dir / 'feature_match_images' / temp_filename
#img_filename = 'jeep-3.jpeg' 
img_filename = 'jeep-1.png'
img_dir = script_dir / 'feature_match_images' / img_filename

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
    '''
    template_rois = {
        "mod_1": [(522, 939), (966, 684)],
        "mod_2": [(681, 151), (925, 590)],
        "mod_3": [(441, 141), (697, 590)],
        "mod_4": [(534, 961), (1003, 1118)],
        "mod_5": [(554, 1204), (1004, 1320)],
        "mod_6": [(990, 1728), (788, 1413)],
        "mod_7": [(808, 1771), (586, 1465)],
        "mod_8": [(546, 1468), (242, 1002)]
    }

    template_rois = {
        "Module_1": [
            -3.390625,
            2.7472527472527473,
            6.9375,
            2.802197802197802
        ],
        "Module_2": [
            -0.90625,
            -3.10989010989011,
            3.8125,
            4.824175824175824
        ],
        "Module_3": [
            -4.65625,
            -3.21978021978022,
            4.0,
            4.934065934065934
        ],
        "Module_4": [
            -3.203125,
            5.791208791208791,
            7.328125,
            1.7252747252747254
        ],
        "Module_5": [
            -2.890625,
            8.461538461538462,
            7.03125,
            1.2747252747252746
        ],
        "Module_6": [
            0.765625,
            10.758241758241759,
            3.15625,
            3.4615384615384617
        ],
        "Module_7": [
            -2.390625,
            11.32967032967033,
            3.46875,
            3.3626373626373627
        ],
        "Module_8": [
            -7.765625,
            6.241758241758242,
            4.75,
            5.1208791208791204
        ]
    }

    colors = [
        (255, 0, 0), #red
        (0, 0, 255), #blue
        (255, 0, 255), #magenta
        (0,255,0), #green
        (255,255,0), #yellow
        (0,255,255), #cyan
        (255,255,255), #white
        (147,112,219) #purple
    ]

    for (roi_name, roi_box), color in zip(template_rois.items(),colors):
        #project this box into the detected image.
        scene_box = transform_box(roi_box, M)

        cv2.polylines(
            img_color,
            [scene_box],
            isClosed=True,
            color=color,
            thickness=4
        )

        #use the first transformed corner for the label.
        label_x, label_y = scene_box[0, 0]

        cv2.putText(
            img_color,
            roi_name,
            (int(label_x), int(label_y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            3,
            cv2.LINE_AA
        )
    
    #display
    cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Object', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if M is not None:
    '''    

    cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Object', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

feature_matching()