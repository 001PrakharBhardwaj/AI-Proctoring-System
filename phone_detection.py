from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])

            label = model.names[cls]

            if label == "cell phone":
                print("Phone Detected!")

    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) == 27:
        break