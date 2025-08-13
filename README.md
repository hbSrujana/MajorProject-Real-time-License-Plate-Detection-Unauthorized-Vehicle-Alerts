# Real-time License Plate Detection & Unauthorized Vehicle Alerts

## 📌 Project Overview
This project implements a smart surveillance system that automatically detects vehicles and their license plates from **video input** using **YOLOv8 object detection** and **Tesseract OCR**. It then cross-verifies detected plates against a **blacklist** stored in a **MySQL database**. If a match is found, it triggers an **audible alert**, a **popup warning message**, and logs the detection in the database.

---

## 🎯 Objectives

### **Objective 1: Vehicle Detection**
- **Goal:** Detect all vehicles in the video frames.
- **Method:**  
  - Uses **YOLOv8 Nano** (pre-trained) for real-time vehicle detection.  
  - Draws bounding boxes and labels (e.g., *Car*, *Truck*) on each detected vehicle.
- **Result:**  
  ![Objective 1 Result](final%20result%20pic/obj1%20result.png)

---

### **Objective 2: License Plate Detection & OCR**
- **Goal:** Detect and read license plates from detected vehicles.
- **Method:**  
  - Uses a **custom-trained YOLOv8 model** for license plate detection.  
  - Crops the detected plate region and processes it using **Tesseract OCR** to extract alphanumeric text.  
  - Cleans extracted text (removes spaces, newlines, converts to uppercase).
- **Result:**  
  ![Objective 2 Result 1](final%20result%20pic/obj2%20res.png)  
  ![Objective 2 Result 2](final%20result%20pic/obj2%20resss.png)  
  ![Objective 2 Result 3](final%20result%20pic/obj2%20result.png)

---

### **Objective 3: Database Integration & Alert System**
- **Goal:** Identify unauthorized vehicles from detected license plates.
- **Method:**  
  - Compares extracted plate numbers with entries in the **MySQL blacklist** table.  
  - If a match is found:  
    - Plays a **beep sound** (Windows `winsound` module).  
    - Displays a **popup warning** using Tkinter.  
    - Logs detection in the `detected_plates` table with timestamp.
- **Result:**  
  ![Objective 3 Result](final%20result%20pic/obj3%20result.png)

---

## 📂 Dataset Description
- **Images Folder:** Contains `.jpg` images of vehicles in different environments.  
- **Labels Folder:** Contains `.txt` files in YOLO format with bounding box coordinates for license plates.  
- **Split:** `train`, `val`, `test` sets for model training and evaluation.

---

## 🛠 Technologies Used
- **Python:** Main programming language for integrating components.
- **YOLOv8 (Ultralytics):** Real-time object detection for vehicles & plates.
- **OpenCV:** For video frame processing.
- **Tesseract OCR:** For reading text from cropped license plate images.
- **MySQL + mysql-connector-python:** Database for blacklist & detected plates logging.
- **Tkinter:** Popup alert GUI.
- **winsound:** Beep alert sound on Windows.

---

## 🚀 How It Works
1. **Vehicle Detection:** YOLOv8 Nano detects vehicles in video frames.
2. **License Plate Detection:** Custom YOLOv8 detects plates within vehicles.
3. **OCR Processing:** Tesseract OCR extracts text from cropped plates.
4. **Blacklist Check:** Compares extracted text with MySQL blacklist.
5. **Alerts & Logging:** If matched, beeps, shows popup, logs in DB.

---

## 📌 Conclusion
This system effectively integrates **real-time object detection**, **OCR**, and **database alerts** to create a reliable unauthorized vehicle monitoring solution.


