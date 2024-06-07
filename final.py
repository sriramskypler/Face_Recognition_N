from flask import Flask, Response, render_template
import cv2
import face_recognition
import numpy as np
import pickle
import os
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor

# Maximum number of encodings to save per face
MAX_ENCODINGS_PER_FACE = 15

# Minimum time interval between face detections (in seconds)
MIN_DETECTION_INTERVAL = 30

# Tolerance for face recognition
TOLERANCE = 0.497

# Define locks for synchronization
known_face_encodings_lock = threading.Lock()
last_known_face_locations_lock = threading.Lock()
last_detection_times_lock = threading.Lock()
unknown_counter_lock = threading.Lock()
last_assigned_id_lock = threading.Lock()

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

# Function to create a directory for each ID and each day
def create_id_date_directory(face_id):
    today_date = datetime.datetime.now().strftime("%Y-%m-%d")
    face_id_directory = os.path.join(today_date, face_id)
    os.makedirs(face_id_directory, exist_ok=True)
    return face_id_directory

# Initialize dictionary to track face movements and last detection times
last_known_face_locations = {}
last_detection_times = {}
unknown_counter = 0  # Initialize unknown counter
last_assigned_id = 0  # Initialize the last assigned ID

# Function to recognize faces and assign IDs
def recognize_faces(frame, known_face_encodings):
    global last_known_face_locations
    global last_detection_times
    global unknown_counter
    global last_assigned_id

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    # Encode the faces found in the frame
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Initialize list to store IDs of recognized faces
    face_ids = []

    # Get the current time
    current_time = datetime.datetime.now()

    # Iterate through each face encoding
    for idx, encoding in enumerate(face_encodings):
        # Initialize face_id
        face_id = None

        # Check if the face encoding is known
        match = False
        for known_id, known_encodings in known_face_encodings.items():
            # Compare the new encoding with known encodings
            match_results = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
            if any(match_results):
                face_id = known_id
                face_ids.append(face_id)
                match = True
                break

        if not match:
            # If face is not known, assign a new ID
            with unknown_counter_lock:
                new_id = f"unknown_{unknown_counter:04d}"
                known_face_encodings[new_id] = [encoding]
                face_ids.append(new_id)
                unknown_counter += 1
                face_id = new_id

        if face_id is not None:
            # Check if enough time has passed since the last save for this face
            last_save_time = last_detection_times.get(face_id)
            if last_save_time is None or (current_time - last_save_time).total_seconds() >= MIN_DETECTION_INTERVAL:
                # Store the new encoding for the face
                with known_face_encodings_lock:
                    if len(known_face_encodings.get(face_id, [])) < MAX_ENCODINGS_PER_FACE:
                        known_face_encodings[face_id].append(encoding)
                    else:
                        known_face_encodings[face_id] = known_face_encodings[face_id][1:] + [encoding]

                # Save the face image to the corresponding directory
                face_id_directory = create_id_date_directory(face_id)
                filename = f"{current_time.strftime('%Y%m%d%H%M%S')}.jpg"
                filepath = os.path.join(face_id_directory, filename)
                top, right, bottom, left = face_locations[idx]

                # Adjust the cropping of the face region with different offsets for each side
                face_image = frame[max(0, top - 150):bottom + 20, max(0, left - 20):right + 20]

                # Convert the cropped face image to BGR before saving to ensure normal color
                bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, bgr_face_image)

                # Update the last detection time for this face
                with last_detection_times_lock:
                    last_detection_times[face_id] = current_time

                # Update the last known location of the face
                with last_known_face_locations_lock:
                    last_known_face_locations[face_id] = face_locations[idx]

    return face_locations, face_ids

# Function to process video frames
def process_frame(frame, known_face_encodings):
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

    return frame

app = Flask(__name__)

# Initialize the ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)

@app.route('/')
def index():
    return render_template('index.html')  # You'll need to create index.html in your templates folder

def generate_frames():
    global known_face_encodings
    rtsp_url = 'rtsp://admin:skypler@sriram@210.18.176.33:554/Streaming/Channels/101'
    video_capture = cv2.VideoCapture(rtsp_url)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        # Process the frame using ThreadPoolExecutor
        future = executor.submit(process_frame, frame, known_face_encodings)
        processed_frame = future.result()

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Load existing face encodings or initialize empty dictionary
    known_face_encodings = load_known_face_encodings()

    # Retrieve the last assigned ID from the pickle file
    if known_face_encodings:
        # Extract the numeric parts of the IDs and find the maximum
        ids = [int(id.split('_')[1]) for id in known_face_encodings.keys() if id.startswith('unknown_')]
        last_assigned_id = max(ids, default=0)
        unknown_counter = last_assigned_id + 1  # Increment unknown_counter from the last assigned ID

    app.run(debug=True)
