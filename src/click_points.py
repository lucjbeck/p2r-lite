import cv2
import json
import sys
VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/Users/lucbeck/Desktop/p2r-lite/data/demo/demo_take1.mov"
OUT = sys.argv[2] if len(sys.argv) > 2 else "points_demo_take1.json"

cap = cv2.VideoCapture(VIDEO)
ok, frame = cap.read()
cap.release()
assert ok, f"could not read first frame of {VIDEO}"

points = []

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((int(x), int(y)))

cv2.namedWindow("click block points")
cv2.setMouseCallback("click block points", on_click)

while True:
    disp = frame.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(disp, (x, y), 4, (0, 255, 0), -1)
        cv2.putText(disp, str(i), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imshow("click block points", disp)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("u") and points:      # undo last click
        points.pop()
    elif key == ord("s"):               # save and exit
        with open(OUT, "w") as f:
            json.dump({"video": VIDEO, "frame": 0, "points": points}, f, indent=2)
        print(f"saved {len(points)} points to {OUT}")
        break
    elif key == ord("q"):               # quit without saving
        print("quit without saving")
        break

cv2.destroyAllWindows()