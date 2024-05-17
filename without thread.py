import cv2
import face_recognition
import numpy as np
import pickle
import os

# Maximum number of encodings to save per face
MAX_ENCODINGS_PER_FACE = 52

# Function to load known face encodings from a pickle file
def load_known_face_encodings():
    if os.path.exists('known_face_encodings.pkl'):
        with open('known_face_encodings.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return {}

# Function to save known face encodings to a pickle file
def save_known_face_encodings(known_face_encodings):
    with open('known_face_encodings.pkl', 'wb') as f:
        pickle.dump(known_face_encodings, f)

# Function to recognize faces and assign IDs
def recognize_faces(frame, known_face_encodings, tolerance=0.6):
    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    # Encode the faces found in the frame
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Initialize list to store IDs of recognized faces
    face_ids = []

    # Iterate through each face encoding
    for encoding in face_encodings:
        # Check if the face encoding is known
        match = False
        for face_id, known_encodings in known_face_encodings.items():
            # Compare the new encoding with known encodings
            if face_recognition.compare_faces(known_encodings, encoding, tolerance=tolerance)[0]:
                face_ids.append(face_id)
                match = True
                break

        if not match:
            # If face is not known, assign a new ID
            new_id = len(known_face_encodings)
            known_face_encodings[new_id] = [encoding]
            face_ids.append(new_id)
        else:
            # Check if the maximum number of encodings is reached for this face ID
            if len(known_encodings) >= MAX_ENCODINGS_PER_FACE:
                continue  # Skip adding more encodings for this face

            # Save the new encoding for the existing face ID
            known_encodings.append(encoding)

    return face_locations, face_ids

# Load existing face encodings or initialize empty dictionary
known_face_encodings = load_known_face_encodings()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 20.0, (640, 480))

# Open video capture
rtsp_url = 'rtsp://admin:skypler@sriram@210.18.176.33:554/Streaming/Channels/101'
video_capture = cv2.VideoCapture(0)
while True:
    # Read a single frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB (required by face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Recognize faces and assign IDs
    face_locations, face_ids = recognize_faces(rgb_frame, known_face_encodings)

    # Draw bounding boxes around faces and label with IDs
    for (top, right, bottom, left), face_id in zip(face_locations, face_ids):
        # Draw bounding box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Label the face with its ID
        cv2.putText(frame, str(face_id), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Write the frame into the video
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save known face encodings to a pickle file
save_known_face_encodings(known_face_encodings)

# Release the video capture object, release the video writer object, and close all windows
video_capture.release()
out.release()
cv2.destroyAllWindows()



