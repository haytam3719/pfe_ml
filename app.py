'''
from google.cloud import vision
import io

def extract_text_from_image(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    return texts[0].description if texts else ""

image_path = 'C:/Users/Haytam/Documents/cin.jpg'
text = extract_text_from_image(image_path)
#print(text)

'''

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from flask import Flask, request, jsonify, Response
import io
from PIL import Image
from collections import deque

# Initialize MTCNN and InceptionResnetV1 models
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Initialize Flask application
app = Flask(__name__)

# Global variable for reference embedding
reference_embedding = None

# Buffer to keep track of match/no match results
decision_buffer = deque(maxlen=20)

# Function to get face embeddings from an image
def get_face_embeddings(image, mtcnn, model):
    try:
        # Detect faces
        faces, probs = mtcnn(image, return_prob=True)
        
        if faces is None:
            raise ValueError("No faces detected.")
        
        embeddings = []
        for i, face in enumerate(faces):
            if probs[i] > 0.90:
                # Get embedding
                with torch.no_grad():
                    embedding = model(face.unsqueeze(0))
                embeddings.append((embedding[0].numpy(), probs[i]))
        
        if embeddings:
            return embeddings
        else:
            raise ValueError("No faces found in the image with sufficient confidence.")
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Function to compare two face embeddings
def compare_faces(embedding1, embedding2, threshold=1.0):
    distance = np.linalg.norm(embedding1 - embedding2)
    print(f"Distance between faces: {distance}")
    return distance < threshold

# Function to make the final decision based on match results
def make_final_decision(buffer):
    if len(buffer) == buffer.maxlen:
        matches = sum(buffer)
        no_matches = len(buffer) - matches
        if matches > no_matches:
            return "Final Decision: Same person"
        else:
            return "Final Decision: Different person"
    return "Decision pending..."

# Endpoint for uploading reference image
@app.route('/upload_reference', methods=['POST'])
def upload_reference():
    global reference_embedding
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    image = Image.open(io.BytesIO(file.read()))
    image_rgb = np.array(image.convert('RGB'))
    
    embeddings_with_probs = get_face_embeddings(image_rgb, mtcnn, model)
    if embeddings_with_probs:
        reference_embedding = embeddings_with_probs[0][0]
        return jsonify({"message": "Reference image uploaded successfully"}), 200
    else:
        return jsonify({"error": "No face detected in the uploaded image"}), 400

# Endpoint for real-time face detection and recognition
@app.route('/video_feed')
def video_feed():
    def generate():
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                embeddings_with_probs = get_face_embeddings(image_rgb, mtcnn, model)
                if embeddings_with_probs:
                    for embedding, prob in embeddings_with_probs:
                        if reference_embedding is not None:
                            match = compare_faces(reference_embedding, embedding)
                            decision_buffer.append(match)
                            if match:
                                cv2.putText(frame, "Match", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            else:
                                cv2.putText(frame, "No Match", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        boxes, _ = mtcnn.detect(image_rgb)
                        if boxes is not None:
                            for box in boxes:
                                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                # Display the decision based on buffer results
                final_decision = make_final_decision(decision_buffer)
                cv2.putText(frame, final_decision, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error during detection: {e}")
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        video_capture.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint to get the final decision
@app.route('/trancher', methods=['GET'])
def trancher():
    final_decision = make_final_decision(decision_buffer)
    return jsonify({"final_decision": final_decision})

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)



# Encode the authorized user's face and save it
#authorized_face_encoding = encode_face("path_to_authorized_user_image.jpg", mtcnn, model)
#np.save("authorized_face_encoding.npy", authorized_face_encoding)




'''
# Analyze face
analysis_result = analyze_face(img_path)
print(f"Age: {analysis_result['age']}")
print(f"Gender: {analysis_result['gender']}")
print(f"Race: {analysis_result['dominant_race']}")
print(f"Emotion: {analysis_result['dominant_emotion']}")

# Detect face and show it
detected_face = detect_faces(img_path)
cv2.imshow("Detected Face", detected_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''