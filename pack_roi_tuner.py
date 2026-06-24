import cv2
import json
import argparse

points = []
module_rois = {}
current_module = 1
image = None
display = None
pack_box = None


def detect_pack_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise RuntimeError("No pack boundary detected.")

    contour = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(contour)


def pixel_roi_to_relative(pack_box, x1, y1, x2, y2):
    px, py, pw, ph = pack_box

    rx = (x1 - px) / pw
    ry = (y1 - py) / ph
    rw = (x2 - x1) / pw
    rh = (y2 - y1) / ph

    return [rx, ry, rw, rh]


def mouse_callback(event, x, y, flags, param):
    global points, current_module, display

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point selected: {x}, {y}")

        cv2.circle(display, (x, y), 5, (0, 255, 255), -1)

        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]

            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])

            module_name = f"Module_{current_module}"

            module_rois[module_name] = pixel_roi_to_relative(
                pack_box,
                x_min,
                y_min,
                x_max,
                y_max
            )

            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                display,
                module_name,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            print(f"Saved {module_name}: {module_rois[module_name]}")

            current_module += 1
            points = []

        cv2.imshow("Click Module ROIs", display)


def main():
    global image, display, pack_box

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--output", default="module_rois.json")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)

    if image is None:
        raise RuntimeError("Could not load image.")

    display = image.copy()

    pack_box = detect_pack_box(image)
    px, py, pw, ph = pack_box

    cv2.rectangle(display, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
    cv2.putText(
        display,
        "Detected Pack Boundary",
        (px, py - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )

    print("Click two corners for each module ROI.")
    print("Example: top-left then bottom-right.")
    print("Press s to save.")
    print("Press u to undo last module.")
    print("Press q to quit.")

    cv2.imshow("Click Module ROIs", display)
    cv2.setMouseCallback("Click Module ROIs", mouse_callback)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            with open(args.output, "w") as f:
                json.dump(module_rois, f, indent=4)

            print(f"Saved ROIs to {args.output}")

        elif key == ord("u"):
            if module_rois:
                last_module = sorted(module_rois.keys())[-1]
                del module_rois[last_module]
                print(f"Removed {last_module}")
                print("Restart script to redraw cleanly.")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()