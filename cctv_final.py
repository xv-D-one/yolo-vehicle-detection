import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import os
import time

class VehicleDetector:
    def __init__(self, model_path):
        """Initialize the vehicle detector with YOLO model"""
        self.model = YOLO(model_path)
        self.unique_cars = set()
        self.unique_bikes = set()
        self.car_count = 0
        self.bike_count = 0
        self.total_detections = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Define vehicle classes (COCO dataset class IDs)
        self.car_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.bike_classes = {1: 'bicycle', 3: 'motorcycle'}
        
    def get_video_source(self):
        """Get video source from user input"""
        print("\n" + "="*50)
        print("🚗 VEHICLE DETECTION SYSTEM 🏍️")
        print("="*50)
        print("Choose video source:")
        print("1. Live webcam stream")
        print("2. Video file")
        print("3. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == '1':
                return 0  # Default webcam
            elif choice == '2':
                while True:
                    video_path = input("Enter video file path: ").strip()
                    if os.path.exists(video_path):
                        return video_path
                    else:
                        print("❌ File not found. Please enter a valid path.")
            elif choice == '3':
                print("👋 Goodbye!")
                exit()
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
    
    def is_vehicle_of_interest(self, class_id):
        """Check if detected object is a car or bike"""
        return class_id in self.car_classes or class_id in self.bike_classes
    
    def classify_vehicle(self, class_id):
        """Classify vehicle as car or bike"""
        if class_id in [2, 5, 7]:  # car, bus, truck
            return 'car'
        elif class_id in [1, 3]:  # bicycle, motorcycle
            return 'bike'
        return None
    
    def generate_unique_id(self, bbox, class_id):
        """Generate a simple unique ID based on position and class"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        # Simple unique ID based on center position (rounded to nearest 50 pixels)
        unique_id = f"{class_id}_{center_x//50}_{center_y//50}"
        return unique_id
    
    def update_counts(self, boxes, frame_shape):
        """Update unique vehicle counts"""
        if boxes is None or len(boxes) == 0:
            return
        
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        
        frame_cars = set()
        frame_bikes = set()
        
        for i in range(len(boxes)):
            class_id = int(cls[i])
            
            if self.is_vehicle_of_interest(class_id):
                bbox = xyxy[i]
                unique_id = self.generate_unique_id(bbox, class_id)
                vehicle_type = self.classify_vehicle(class_id)
                
                if vehicle_type == 'car':
                    frame_cars.add(unique_id)
                    if unique_id not in self.unique_cars:
                        self.unique_cars.add(unique_id)
                        self.car_count += 1
                elif vehicle_type == 'bike':
                    frame_bikes.add(unique_id)
                    if unique_id not in self.unique_bikes:
                        self.unique_bikes.add(unique_id)
                        self.bike_count += 1
    
    def draw_detections(self, frame, result):
        """Draw bounding boxes and labels on frame"""
        frame_labeled = frame.copy()
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy()
            
            class_names = self.model.names
            
            for i in range(len(boxes)):
                class_id = int(cls[i])
                
                # Only draw vehicles of interest
                if self.is_vehicle_of_interest(class_id):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    confidence = conf[i]
                    class_name = class_names[class_id]
                    
                    # Choose color based on vehicle type
                    vehicle_type = self.classify_vehicle(class_id)
                    if vehicle_type == 'car':
                        color = (0, 255, 0)  # Green for cars
                        label = f"🚗 {class_name}: {confidence:.2f}"
                    else:
                        color = (255, 0, 0)  # Blue for bikes
                        label = f"🏍️ {class_name}: {confidence:.2f}"
                    
                    # Draw bounding box
                    cv2.rectangle(frame_labeled, (x1, y1), (x2, y2), color, 2)
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Draw background rectangle for text
                    cv2.rectangle(
                        frame_labeled,
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        color,
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        frame_labeled,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        return frame_labeled
    
    def draw_info_panel(self, frame):
        """Draw information panel on frame"""
        # Create info panel background
        panel_height = 120
        cv2.rectangle(frame, (10, 10), (400, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, panel_height), (255, 255, 255), 2)
        
        # Add text information
        info_texts = [
            f"Frame: {self.frame_count}",
            f"🚗 Cars Detected: {self.car_count}",
            f"🏍️ Bikes Detected: {self.bike_count}",
            f"Total Vehicles: {self.car_count + self.bike_count}"
        ]
        
        for i, text in enumerate(info_texts):
            y_pos = 35 + i * 20
            cv2.putText(
                frame,
                text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    
    def print_summary(self):
        """Print final detection summary"""
        duration = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("🚗 VEHICLE DETECTION SUMMARY 🏍️")
        print("="*60)
        print(f"📊 Total Frames Processed: {self.frame_count}")
        print(f"⏱️  Processing Duration: {duration:.2f} seconds")
        print(f"🎯 Average FPS: {self.frame_count/duration:.2f}")
        print(f"🚗 Unique Cars Detected: {self.car_count}")
        print(f"🏍️  Unique Bikes Detected: {self.bike_count}")
        print(f"🔢 Total Unique Vehicles: {self.car_count + self.bike_count}")
        print("="*60)
        
        if self.car_count > 0 or self.bike_count > 0:
            print("✅ Detection completed successfully!")
        else:
            print("⚠️  No vehicles detected in the video.")
        print("="*60)
    
    def run_detection(self):
        """Main detection loop"""
        # Get video source from user
        video_source = self.get_video_source()
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("❌ Error: Could not open video source")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n📹 Video properties: {width}x{height} at {fps} FPS")
        print("🎮 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame")
        print("   - Press 'p' to pause/resume")
        print("\n🚀 Starting detection...")
        
        # Optional: Set up video writer
        save_output = False
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('vehicle_detection_output.avi', fourcc, fps, (width, height))
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("📺 End of video or failed to read frame")
                        break
                    
                    self.frame_count += 1
                    
                    # Make predictions
                    results = self.model(frame, verbose=False)
                    result = results[0]
                    
                    # Update counts
                    self.update_counts(result.boxes, frame.shape)
                    
                    # Draw detections
                    frame_labeled = self.draw_detections(frame, result)
                    
                    # Draw info panel
                    self.draw_info_panel(frame_labeled)
                    
                    # Display frame
                    cv2.imshow('🚗 Vehicle Detection System 🏍️', frame_labeled)
                    
                    # Save frame if output enabled
                    if save_output:
                        out.write(frame_labeled)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f'vehicle_detection_frame_{self.frame_count}.jpg', frame_labeled)
                    print(f"💾 Saved frame {self.frame_count}")
                elif key == ord('p'):
                    paused = not paused
                    print("⏸️ Paused" if paused else "▶️ Resumed")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            # Clean up
            cap.release()
            if save_output:
                out.release()
            cv2.destroyAllWindows()
            
            # Print summary
            self.print_summary()

def main():
    """Main function"""
    # Model path - update this to your model path
    model_path = r"runs\detect\train\weights\best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please update the model_path variable with the correct path to your YOLO model.")
        return
    
    # Create detector instance
    detector = VehicleDetector(model_path)
    
    # Run detection
    detector.run_detection()

if __name__ == "__main__":
    main()