import cv2
import numpy as np
import os
from deepface import DeepFace
import pygame

pygame.mixer.init()

# Get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Emotion image paths (absolute paths)
emotion_images = {
    'angry': os.path.join(BASE_DIR, 'imgs', 'angry.png'),
    'disgust': os.path.join(BASE_DIR, 'imgs', 'disgust.png'),
    'fear': os.path.join(BASE_DIR, 'imgs', 'fear.png'),
    'happy': os.path.join(BASE_DIR, 'imgs', 'happy.png'),
    'sad': os.path.join(BASE_DIR, 'imgs', 'sad.png'),
    'surprise': os.path.join(BASE_DIR, 'imgs', 'surprise.png'),
    'neutral': os.path.join(BASE_DIR, 'imgs', 'neutral.png')
}

# Emotion music paths (absolute paths)
emotion_music = {
    'angry': os.path.join(BASE_DIR, 'music', 'angry.mp3'),
    'disgust': os.path.join(BASE_DIR, 'music', 'disgust.mp3'),
    'fear': os.path.join(BASE_DIR, 'music', 'fear.mp3'),
    'happy': os.path.join(BASE_DIR, 'music', 'happy.mp3'),
    'sad': os.path.join(BASE_DIR, 'music', 'sad.mp3'),
    'surprise': os.path.join(BASE_DIR, 'music', 'surprise.mp3')
}

# Load emotion images ONCE
emotion_images_loaded = {}
for emotion, path in emotion_images.items():
    img = cv2.imread(path)
    if img is None:
        print(f"Error loading image: {path}")
    else:
        emotion_images_loaded[emotion] = cv2.resize(img, (300, 300))

# Playing voice
def play_music(file_path):
    if os.path.exists(file_path):
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(-1)  # Loop indefinitely
    else:
        print(f"Error: Music file not found at {file_path}")


def stop_music():
    pygame.mixer.music.stop()


# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize variables to track the current emotion
current_emotion = None
music_playing = False
frame_count = 0
analysis_interval = 5  # Analyze every N frames


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)  # For DeepFace

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        frame_count += 1
        if frame_count % analysis_interval == 0:  # Analyze every N frames
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            for result in results:
                # Determine the dominant emotion
                emotion = result['dominant_emotion']

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"{emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Load and display emotion image
                emotion_img = emotion_images_loaded.get(emotion)  # Retrieve pre-loaded image

                # Handle music playback
                if emotion in emotion_music:
                    music_path = emotion_music[emotion]
                    if emotion != current_emotion:
                        stop_music()
                        play_music(music_path)
                        current_emotion = emotion
                        music_playing = True  # Set flag
                    elif emotion == current_emotion and not music_playing: #if the music is not playing for the same emotion
                        play_music(music_path)
                        music_playing = True
                else:
                    stop_music()
                    current_emotion = None
                    music_playing = False  # Reset flag

                # Display the resulting frame

                if emotion_img is not None:
                    cv2.imshow('Emotion Image', emotion_img)

    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit() # Close the mixer