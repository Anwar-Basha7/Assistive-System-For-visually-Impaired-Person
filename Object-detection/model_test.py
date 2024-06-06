import cv2
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('C:/Users/USER/yolov8_env/Object-detection/best.pt')

def detect_objects(frame):
    # Resize the frame to 640x640
    resized_frame = cv2.resize(frame, (640, 640))
    # Perform detection
    results = model.predict(source=resized_frame)
    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0]
            # Draw the bounding box
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put the label and confidence score
            cv2.putText(resized_frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return resized_frame

def main():
    # Open the camera
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the live frame
        cv2.imshow('Press Space to Capture Image', frame)
        
        # Capture the image on pressing the space bar
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Detect objects in the captured frame
            detected_frame = detect_objects(frame)
            
            # Display the frame with detections
            cv2.imshow('YOLOv8 Object Detection', detected_frame)
            
            # Wait for any key press to continue
            cv2.waitKey(0)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()