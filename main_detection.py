import cv2  # OpenCV for video and image processing
import pytesseract  # For Optical Character Recognition (OCR)
from ultralytics import YOLO # For loading and using YOLOv8 models
import mysql.connector # To connect to MySQL database
from datetime import datetime # To get current time for logging
import winsound        # To play beep sounds (Windows only)
from tkinter import Tk, messagebox  # For popup alert messages

# Specifies the path where Tesseract OCR is installed on your system.
# Why it's needed: If this path isn't set, Tesseract won't work and OCR will fail.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO models
#This is the pretrained base model downloaded automatically from Ultralytics during training. It's YOLOv8 nano version, optimized for lightweight performance.
vehicle_model = YOLO("yolov8n.pt")
plate_model = YOLO("LicensePlateTraining/yolov8n_custom_test2/weights/best.pt")

# MySQL connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Srujana$#@67",  # change it to your actual password
    database="vehicle_db"
)
cursor = conn.cursor()
cursor.execute("SELECT plate FROM blacklist")
blacklist = set(row[0].strip().upper() for row in cursor.fetchall())

# Output video setup
video_path = "input1_video.mp4"
cap = cv2.VideoCapture(video_path)
output_size = (960, 540)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("final_output.mp4", fourcc, fps, output_size)

# Previous plates memory
prev_plates = []

# Plates already alerted in this run
alerted_plates = set()

# Alert popup setup
root = Tk()
root.withdraw()

# IoU for OCR stability
def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    vehicle_results = vehicle_model(frame, conf=0.4)[0]
    plate_results = plate_model(frame, conf=0.4)[0]
    current_plates = []

    # Draw vehicle boxes
    for box in vehicle_results.boxes:
        cls_id = int(box.cls[0])
        label = vehicle_model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 2)

    # Detect and OCR plates
    for box in plate_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        matched_text = None

        # OCR text reuse using IoU
        for prev_box, prev_text in prev_plates:
            if compute_iou((x1, y1, x2, y2), prev_box) > 0.5:
                matched_text = prev_text
                break

        # If not matched, run OCR
        if not matched_text:
            matched_text = pytesseract.image_to_string(gray, config='--psm 7').strip()

        normalized_text = matched_text.upper().replace(" ", "").replace("\n", "")
        current_plates.append(((x1, y1, x2, y2), normalized_text))

        # Draw plate and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        if normalized_text:
            cv2.putText(frame, normalized_text, (x1, y2 + 35), cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 0), 4)

            # Check and alert once per blacklisted plate
            if normalized_text in blacklist and normalized_text not in alerted_plates:
                winsound.Beep(1000, 300)
                messagebox.showwarning("Alert", f"Blacklisted Plate Detected: {normalized_text}")
                cursor.execute("INSERT INTO detected_plates (plate, detected_time) VALUES (%s, %s)",
                               (normalized_text, datetime.now()))
                conn.commit()
                alerted_plates.add(normalized_text)

    prev_plates = current_plates

    # Resize and save frame
    resized = cv2.resize(frame, output_size)
    out.write(resized)
    cv2.imshow("Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
conn.close()

print(" Final video saved: final_output.mp4")
