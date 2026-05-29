import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection()

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            h, w, c = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            cv2.rectangle(frame, (x, y),
                          (x + width, y + height),
                          (0, 255, 0), 2)

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break