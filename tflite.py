# import cv2
# import numpy as np
# import tensorflow as tf

# # Load the TensorFlow Lite model
# interpreter = tf.lite.Interpreter(model_path="face_detection_model.tflite")
# interpreter.allocate_tensors()

# # Get input and output details
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# # Open the video capture
# rtsp_url = 'rtsp://admin:skypler@sriram@210.18.176.33:554/Streaming/Channels/101'
# cap = cv2.VideoCapture(rtsp_url)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     # Preprocess the frame
#     input_data = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
#     input_data = np.expand_dims(input_data, axis=0)
#     input_data = (np.float32(input_data) - 127.5) / 127.5  # Normalize

#     # Set the input tensor
#     interpreter.set_tensor(input_details[0]['index'], input_data)

#     # Run inference
#     interpreter.invoke()

#     # Get the output
#     output_data = interpreter.get_tensor(output_details[0]['index'])

#     # Postprocess the output (if needed) and draw on the frame
#     # Replace this with your own postprocessing and drawing logic

#     # Display the frame
#     cv2.imshow('Video', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close OpenCV windows
# cap.release()
# cv2.destroyAllWindows()


#the above code is without flask and below code is with flask. both are just used to Streaming
#not for detect or recognize

from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="face_detection_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to capture video feed from RTSP and perform inference
def generate_frames():
    rtsp_url = 'rtsp://admin:skypler@sriram@210.18.176.33:554/Streaming/Channels/101'
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        input_data = cv2.resize(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        input_data = np.expand_dims(input_data, axis=0)
        input_data = (np.float32(input_data) - 127.5) / 127.5  # Normalize

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Postprocess the output (if needed) and draw on the frame
        # Replace this with your own postprocessing and drawing logic

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in the multipart HTTP response format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

