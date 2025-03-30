import kagglehub
import cv2
import face_recognition
import numpy as np
from deepface import DeepFace
import os
import csv
from datetime import datetime
from collections import defaultdict

# Step 1: Download the FER-2013 dataset
path = kagglehub.dataset_download("msambare/fer2013")
print("Path to dataset files:", path)

# Initialize webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Set video resolution for better performance
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Prepare to save the emotion data to a CSV file
csv_filename = 'emotion_predictions.csv'
fieldnames = ['Timestamp', 'Predicted Emotion', 'Emotion Frequency']

# Create or open the CSV file
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Dictionary to track emotion frequencies
    emotion_freq = defaultdict(int)

    print("Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame from webcam. Exiting...")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)

        # Emotion Detection using DeepFace
        try:
            emotion_analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(emotion_analysis, list):
                emotions = [analysis.get('dominant_emotion', 'Unknown') for analysis in emotion_analysis]
            else:
                emotions = [emotion_analysis.get('dominant_emotion', 'Unknown')] * len(face_locations)
        except Exception as e:
            print(f"Emotion detection failed: {e}")
            emotions = ["Unknown"] * len(face_locations)

        # Update the emotion frequencies
        for emotion in emotions:
            emotion_freq[emotion] += 1

        # Draw rectangles and labels around detected faces
        for (top, right, bottom, left), emotion in zip(face_locations, emotions):
            # Draw rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Prepare text
            text = f"Emotion: {emotion}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Calculate text width and height
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Draw background rectangle for text
            cv2.rectangle(frame, (left, bottom - text_height - 10), (left + text_width + 10, bottom), (0, 255, 0), cv2.FILLED)

            # Draw text
            cv2.putText(frame, text, (left + 5, bottom - 5), font, font_scale, (0, 0, 0), thickness)

        # Save the emotion data into the CSV file
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Write the emotion data to CSV for each emotion detected
        for emotion in emotions:
            writer.writerow({'Timestamp': timestamp, 
                             'Predicted Emotion': emotion, 
                             'Emotion Frequency': emotion_freq[emotion]})

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
