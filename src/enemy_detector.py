from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def detect_enemy(frame):

    results = model(frame)

    enemies = []

    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            if cls == 0:   # person class

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                w = x2 - x1
                h = y2 - y1

                enemies.append((x1,y1,w,h))

    return enemies