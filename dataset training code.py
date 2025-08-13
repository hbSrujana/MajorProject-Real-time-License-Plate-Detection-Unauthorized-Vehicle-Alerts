from ultralytics import YOLO # Import YOLO from ultralytics
model = YOLO("yolov8n.pt")  # Load the pretrained YOLOv8 Nano model
model.train(  #Starts the training process
    data="C:/Users/hbsru/OneDrive/Pictures/Documents/personal/Major project/Major project code/dataset3/data.yaml",# Path to YAML file defining class names and data split
    epochs=3,  # Number of training rounds over entire dataset
    imgsz=640,  # Input image size
    batch=8,     # Batch size (number of images per training step)
    project="LicensePlateTraining",  # Project folder name
    name="yolov8n_custom_test" # Subfolder for saving weights and results
)


#Training Output:
#best.pt: Best performing model based on validation accuracy.
#last.pt: Final model from last epoch.
#These are stored in LicensePlateTraining/yolov8n_custom_test/weights/

#code:
# 1) from ultralytics import YOLO 
# Import YOLO from ultralytics
#Ultralytics is the official library that supports YOLOv8. It provides simple methods for training, inference, and model handling.

#2)model = YOLO("yolov8n.pt")
# What it does:
#This loads the pre-trained YOLOv8n (Nano) model.
#What is yolov8n.pt:
#yolov8n stands for YOLOv8-Nano.
#It’s the lightest and fastest version of YOLOv8, suitable for quick training and testing.
#.pt means it’s a PyTorch model file.
#  Why this line:
# Instead of training from scratch (which needs lots of data and time), we start from a pre-trained model and fine-tune it on our dataset.

#3)model.train(
# Starts the training process for the YOLOv8 model.
# data="C:/Users/hbsru/OneDrive/Pictures/Documents/personal/Major project/Major project code/dataset3/data.yaml",
#  What it does:
# This is the path to the data.yaml file.
#  What is in data.yaml:
# A YAML configuration file that defines:(YAML stands for "YAML Ain’t Markup Language" — it's a simple, human-readable file format often used for configuration.)
# In your project, the YAML file (data.yaml) is a configuration file that tells the YOLOv8 model:
# Where your training, validation, and test images are located
# What the class names are (e.g., license_plate)
# How many object classes (nc) there are
# Why needed:
# YOLO needs to know:
# Where the images are
# What the labels are
# How many object classes are present

#4) epochs=3,
#  What it does:
# This sets the number of training epochs.
#  What is an epoch?
# One full pass over the entire training dataset.
#  Why 3 epochs:
# Just a quick test run (more epochs would improve accuracy). For better results, usually 50–100+ epochs are used.

#5)imgsz=640,
#  What it does:
# Sets the image size to 640×640 pixels for training.
# Why 640:
# YOLO models are trained on square images. 640 is a standard size that balances performance and speed.

#6)batch=8,
#  What it does:
# Sets the batch size — number of images processed together in one step.
#  Why batch size matters:
# Smaller batch size = less memory usage, slower training
# Larger batch size = more memory usage, faster training (if GPU supports)

# project="LicensePlateTraining",
#  What it does:
# Creates a project folder named LicensePlateTraining.
#  Why:
# All outputs like model weights, plots, metrics, and logs are stored inside this folder.
#     name="yolov8n_custom_test"
#  What it does:
# Within the project folder, it creates a subfolder named yolov8n_custom_test to store results of this specific training run.

# After Training — What Gets Created?
# LicensePlateTraining/
# └── yolov8n_custom_test/
#     ├── weights/
#     │   ├── best.pt   ← Best model during validation
#     │   └── last.pt   ← Final model after last epoch
#     ├── results.png   ← Accuracy/loss curve
#     ├── confusion_matrix.png
#     └── ...
# best.pt = Best model (based on validation mAP)
# last.pt = Model at the end of training