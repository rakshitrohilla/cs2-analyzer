import cv2
import math
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

video_path = "video/match1.mp4"

cap = cv2.VideoCapture(video_path)

good_aim = 0
total_checks = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width, _ = frame.shape

    cx = width // 2
    cy = height // 2

    results = model(frame)

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            head_x = x1 + w // 2
            head_y = y1 + int(h * 0.2)

            distance = math.sqrt((head_x - cx)**2 + (head_y - cy)**2)

            # engagement zone
            engagement_radius = 350

            screen_distance = math.sqrt((head_x - cx)**2 + (head_y - cy)**2)

            if screen_distance > engagement_radius:
                continue

            total_checks += 1

            threshold = h * 1.2

            if distance < threshold:
                good_aim += 1
                color = (0,255,0)
                status = "GOOD"
            else:
                color = (0,0,255)
                status = "MISS"

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

            cv2.circle(frame,(head_x,head_y),5,(255,0,0),-1)

            cv2.circle(frame,(cx,cy),6,(0,255,255),-1)

            cv2.line(frame,(cx,cy),(head_x,head_y),(255,255,0),1)

            cv2.putText(frame,f"{status} {int(distance)}",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(255,255,255),1)

    cv2.imshow("CS2 Aim Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if total_checks > 0:
    score = (good_aim / total_checks) * 100
else:
    score = 0

print("Aim Accuracy:", round(score,2), "%")