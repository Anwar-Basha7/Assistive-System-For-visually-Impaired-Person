import speech_recognition as sr

# Initialize recognizer and specify the microphone device index
r = sr.Recognizer()
mic_index = 1  # Try changing this to other values if 3 doesn't work

try:
    # Attempt to open the microphone
    with sr.Microphone(device_index=mic_index) as source:
        print("Say something!")
        # Adjust for ambient noise to improve recognition accuracy
        r.adjust_for_ambient_noise(source)
        # Listen for the first phrase spoken
        audio = r.listen(source)
        print("Got it! Now to recognize it...")
        try:
            # Recognize speech using Google's web service
            command = r.recognize_google(audio)
            print(f"You said: {command}")
        except sr.RequestError as e:
            # API was unreachable or unresponsive
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except sr.UnknownValueError:
            # Speech was unintelligible
            print("Could not understand audio")
except sr.RequestError as e:
    # Network error
    print(f"Network error: {e}")
except sr.UnknownValueError:
    # Microphone index or other audio issues
    print("Could not understand audio")
except Exception as e:
    # Catch any other exceptions
    print(f"Error: {e}")
