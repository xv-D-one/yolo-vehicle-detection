# YOLO Vehicle Detection System

Real-time vehicle detection system using YOLO to detect and count cars and bikes with live visual feedback.

## Introduction
A vehicle detection system that can automatically identify and count cars and bikes in real-time video streams or recorded videos. The system uses YOLO (You Only Look Once), which is a powerful deep learning model for object detection. The main goal was to create a practical application that can monitor traffic by detecting vehicles and keeping track of unique counts without counting the same vehicle multiple times.

## Project Objectives
- Build a system that can detect cars and bikes accurately
- Implement unique counting to avoid counting the same vehicle twice
- Make it work with both webcam and video files
- Show live detection results with visual feedback
- Generate a final summary report

## Technical Implementation

### Dependencies
- OpenCV: For video processing and display
- Ultralytics YOLO: For the detection model
- NumPy: For numerical calculations

### Key Features
- Unique vehicle identification system
- Visual feedback with color-coded bounding boxes
  - Green boxes for cars
  - Blue boxes for bikes
- Confidence scores for each detection
- Live count display in the corner
- Real-time notifications when new vehicles are detected

### Main Functions
```python
class CarBikeDetector:
    def get_video_source()    # Lets user choose between webcam or video file
    def generate_unique_id()  # Creates unique identifiers for vehicles
    def update_counts()       # Updates the vehicle counters
    def detect_vehicles()     # Main detection loop
    def print_summary()       # Shows final results
```

## Detection Process
The system works by taking each frame from the video and running it through the YOLO model. YOLO looks at the entire image at once and identifies objects along with their locations and confidence scores.

### Vehicle Types Detected
- Cars: Including regular cars, buses, and trucks
- Bikes: Including bicycles and motorcycles

### Unique Counting Method
To solve the problem of counting the same vehicle multiple times, I created a unique identification system. For each detected vehicle, I generate a unique ID based on:
- The vehicle's center position in the frame
- The type of vehicle (car or bike)
- Rounding the position to a grid to handle small movements

This way, even if the same car appears in multiple consecutive frames, it only gets counted once.

## Usage
1. User chooses video source (webcam or file)
2. Detection starts automatically
3. Live results shown on screen
4. Press 'q' to quit and see summary

## Learning Outcomes
- Computer vision and object detection
- Working with pre-trained deep learning models
- Real-time video processing
- Python programming with OpenCV
- Problem-solving for practical applications

## Conclusion
I successfully created a vehicle detection system that meets all the original objectives. The system can accurately detect and count cars and bikes in real-time, providing a useful tool for traffic monitoring applications.

![Screenshot 2025-06-20 105647](https://github.com/user-attachments/assets/cb9030e5-472b-4beb-b33c-457b4ef072f5)
![Screenshot 2025-06-20 110117](https://github.com/user-attachments/assets/903b206d-83f1-458e-8373-807462696a88)
![Screenshot 2025-06-20 110330](https://github.com/user-attachments/assets/650d953a-30bf-4ad1-8c24-7acb6f8e17cc)
![Screenshot 2025-06-20 110403](https://github.com/user-attachments/assets/00792b29-66a8-4806-bdfa-7672260109bf)
![Screenshot 2025-06-20 110810](https://github.com/user-attachments/assets/50122f33-753c-46f1-b358-03a74a65e8ce)





