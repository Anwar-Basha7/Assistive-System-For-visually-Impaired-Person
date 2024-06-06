import speech_recognition as sr
import cv2
import pytesseract
from PIL import Image
import os
from ultralytics import YOLO
import pyttsx3
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import pickle
import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the trained YOLOv8 model
model = YOLO('C:/Users/USER/yolov8_env/Object-detection/best.pt')

# Function to detect objects in the frame
def detect_objects(frame):
    resized_frame = cv2.resize(frame, (640, 640))
    results = model.predict(source=resized_frame)
    detected_objects = {}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]

            if label in detected_objects:
                detected_objects[label] += 1
            else:
                detected_objects[label] = 1

            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return resized_frame, detected_objects

# Function to announce detected objects
def announce_objects(objects):
    print("In announcement function")
    print(objects)
    if not objects:
        engine.say("No objects detected.")
    else:
        for obj in objects:
            announcement = f"Detected {objects[obj]} {obj}(s)"
            print(announcement)
            engine.say(announcement)
    engine.runAndWait()

# Main function to capture and process images for object detection
def capture_and_process():
    cap = VideoStream(src=0).start()
    time.sleep(2.0)

    fps = FPS().start()

    while True:
        frame = cap.read()
        if frame is None:
            print("Error: Could not read frame.")
            break

        detected_frame, detected_objects = detect_objects(frame)
        announce_objects(detected_objects)
        cv2.imshow('YOLOv8 Object Detection', detected_frame)

        start_time = time.time()
        while time.time() - start_time < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.stop()
        cv2.destroyAllWindows()
        break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Function to detect faces
def start_face_recognition():
    currentname = "unknown"
    encodingsP = "encodings.pickle"

    print("[INFO] loading encodings + face detector...")
    data = pickle.loads(open(encodingsP, "rb").read())

    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    fps = FPS().start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        boxes = face_recognition.face_locations(frame)
        encodings = face_recognition.face_encodings(frame, boxes)
        names = []

        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

                if currentname != name:
                    currentname = name
                    print(currentname)

            names.append(name)

        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2)

        cv2.imshow("Facial Recognition is Running", frame)
        key = cv2.waitKey(1) & 0xFF
        start_time = time.time()
        while time.time() - start_time < 3:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        break

        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()
    return names

# Announcing the detected faces
def announce_names(names):
    print("In announcement function")
    print(names)
    if not names:
        engine.say("No faces detected.")
    else:
        for name in names:
            announcement = f"Detected {name}"
            print(announcement)
            engine.say(announcement)
    engine.runAndWait()

#speech to text function starts here
def speech_to_text():
    key = cv2. waitKey(1)
    webcam = cv2.VideoCapture(0)
    while True:
        try:
            check, frame = webcam.read()
            cv2.imshow("Capturing", frame)
            key = cv2.waitKey(1)
            if key == ord('z'):
                cv2.imwrite(filename='saved_img.jpg', img=frame)
                webcam.release()
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                string = pytesseract.image_to_string('saved_img.jpg')
                print(string)
                engine.setProperty('rate', 125) 
                engine.say("hi")
                engine.say(string)
                engine.runAndWait()
                print("Image saved!")
                cv2.destroyAllWindows()
                break
                
            
        except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    
# Function to recognize speech and trigger object detection or face recognition
def recognize_speech_and_trigger():
    r = sr.Recognizer()
    mic_index = 1  # Try changing this to other values if 1 doesn't work

    while True:
        try:
            # Attempt to open the microphone
            with sr.Microphone(device_index=mic_index) as source:
                print("Say 'surrounding' to detect objects!")
                print("Say 'detect face' to detect the persons in front of you")
                print("say 'text to speech' to trigger the text to speech model")
                engine.say("Say 'surrounding' to detect objects!")
                engine.runAndWait()
                engine.say("Say 'detect face' to detect the persons in front of you")
                engine.runAndWait()
                # Adjust for ambient noise to improve recognition accuracy
                r.adjust_for_ambient_noise(source)
                # Listen for the first phrase spoken
                audio = r.listen(source)

            try:
                # Recognize speech using Google's web service
                command = r.recognize_google(audio)
                print(f"Command received: {command}")

                if command.lower() == "surrounding":
                    capture_and_process()
                elif command.lower() == "detect face":
                    names = start_face_recognition()
                    announce_names(names)
                elif command.lower() == "text to speech":
                    speech_to_text()
                else:
                    print("Invalid command. Please say 'surrounding' or 'detect face'.")
                    engine.say("Invalid command. Please say 'surrounding' or 'detect face'.")
                    engine.runAndWait()

            except sr.RequestError as e:
                # API was unreachable or unresponsive
                print(f"Could not request results from Google Speech Recognition service; {e}")
                engine.say("Could not request results from Google Speech Recognition service")
                engine.runAndWait()
            except sr.UnknownValueError:
                # Speech was unintelligible
                print("Could not understand audio")
                engine.say("Could not understand audio")
                engine.runAndWait()
        except sr.RequestError as e:
            # Network error
            print(f"Network error: {e}")
            engine.say("Network error")
            engine.runAndWait()
        except sr.UnknownValueError:
            # Microphone index or other audio issues
            print("Could not understand audio")
            engine.say("Could not understand audio")
            engine.runAndWait()
        except Exception as e:
            # Catch any other exceptions
            print(f"Error: {e}")

if __name__ == '__main__':
    recognize_speech_and_trigger()
