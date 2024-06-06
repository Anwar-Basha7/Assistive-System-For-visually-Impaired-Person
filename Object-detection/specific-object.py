import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('C:/Users/pk/yolov8_env/Object-detection/visionbest.pt')

def getObjects(img, objects=[], draw=True):
    # Resize the frame to 640x640
    resized_frame = cv2.resize(img, (640, 640))
    # Perform detection
    results = model(resized_frame)
    objectInfo = []
    class_names = model.names  # Access the class names from the model
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            if class_name in objects:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                objectInfo.append(([x1, y1, x2, y2], class_name))
                if draw:
                    # Draw the bounding box
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Put the label and confidence score
                    cv2.putText(resized_frame, f"{class_name.upper()} {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return resized_frame, objectInfo

def main():
    # Open the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Ask the user for the target class
    target_class = input("Enter the object class to detect (e.g., 'chair'): ").strip()
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the live frame
        cv2.imshow('Live Camera Feed - Press Space to Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Capture the image on pressing the space bar
        if key == ord(' '):
            # Detect objects in the captured frame
            detected_frame, objectInfo = getObjects(frame, objects=[target_class])
            
            # Display the frame with detections
            cv2.imshow('YOLOv8 Object Detection', detected_frame)
            
            # Wait for any key press to continue
            cv2.waitKey(0)
        
        # Exit on pressing 'q'
        if key == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
