import cv2
import time
import pyttsx3
from ultralytics import YOLO

#video_path = r"G:\project code january\object_detection_distance\bus.jpg"
model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

engine = pyttsx3.init()

# Parameters
FOCAL_LENGTH = 360
KNOWN_WIDTH = 60
INTERVAL = 5
prev = 0

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
my_file.close()

detection_colors = [(0, 255, 0)] * len(class_list)


def perform_object_detection(frame, confidence_threshold=0.8):
    results = model(frame,show=True)

    if results and any(res.boxes for res in results):
        filtered_detections = []
        for res in results:
            for bbox in res.boxes:
                if bbox.conf >= confidence_threshold:
                    filtered_detections.append(res)
                    break

        if filtered_detections:
            max_area = 0
            index = 0
            for i, res in enumerate(filtered_detections):
                for j, bbox in enumerate(res.boxes):
                    area = bbox.xyxy[0][2] * bbox.xyxy[0][3]
                    if area > max_area:
                        max_area = area
                        index = i

            return filtered_detections[index]
    return None

def annotate_frame(frame, detections):
    for detection in detections:
        bbox = detection.boxes[0]
        class_id = int(bbox.cls)
        class_name = detection.names[class_id]
        x1, y1, x2, y2 = bbox.xyxy[0]

        width_pixels = x2 - x1
        distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / width_pixels

        print(f"Detected {class_name} at {distance_cm:.2f} cm")
        if distance_cm < 50:

            speech_text = f"Detected {class_name} at {distance_cm:.2f} centimeters. "
        else:

            speech_text = f"Detected {class_name} at {distance_cm:.2f} centimeters."

        engine.say(speech_text)
        engine.runAndWait()

        color = detection_colors[class_id]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, f"{class_name} {distance_cm:.2f}cm", (int(x1), int(y1) - 10), font, 1, color, 2)

while True:
    _, frame = cap.read()

    if time.time() - prev >= INTERVAL:
        detection = perform_object_detection(frame)

        if detection:
            annotate_frame(frame, detection)
        else:
            print("No detection")

        prev = time.time()

    cv2.imshow("Object Detection and Annotation", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
