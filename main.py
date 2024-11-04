import cv2
import numpy as np
from ultralytics import YOLO
from vidgear.gears import CamGear
import cvzone
from polym import PolylineManager

stream = CamGear(source='https://www.youtube.com/watch?v=_TusTf0iZQU', stream_mode=True, logging=True).start()

with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

model = YOLO("yolo11s.pt")

polyline_manager = PolylineManager()
cv2.namedWindow('RGB')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        polyline_manager.add_point((x, y))

cv2.setMouseCallback('RGB', RGB)

going_up = {}
going_down = {}
gnu = []
gnd = []
while True:
    frame = stream.read()

    frame = cv2.resize(frame, (1020, 500))
    results = model.track(frame, persist=True, classes=[2])

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if polyline_manager.point_polygon_test((cx, cy), 'area1'):
                going_up[track_id] = (cx, cy)
            if track_id in going_up:
                if polyline_manager.point_polygon_test((cx, cy), 'area2'):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                    if gnu.count(track_id) == 0:
                        gnu.append(track_id)

            if polyline_manager.point_polygon_test((cx, cy), 'area2'):
                going_down[track_id] = (cx, cy)
            if track_id in going_down:
                if polyline_manager.point_polygon_test((cx, cy), 'area1'):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                    cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                    if gnd.count(track_id) == 0:
                        gnd.append(track_id)

    godown = len(gnd)
    goup = len(gnu)
    cvzone.putTextRect(frame, f'GoDown: {godown}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'GoUp: {goup}', (50, 160), 2, 2)

    frame = polyline_manager.draw_polylines(frame)

    cv2.imshow("RGB", frame)

    if not polyline_manager.handle_key_events():
        break

stream.stop()
cv2.destroyAllWindows()
