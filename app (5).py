import flask
from flask import Flask, request, jsonify
import os
import pickle
import face_recognition
import numpy as np
from io import BytesIO
import time
from sklearn.neighbors import KNeighborsClassifier

# --- Global Constants ---
VOTER_IMAGE_DIR = 'voter_images'
ENCODINGS_FILE = 'voter_encodings.pkl'
MODEL_FILE = 'voter_face_recognition_model.pkl'

# Create the directory if it doesn't exist
if not os.path.exists(VOTER_IMAGE_DIR):
    os.makedirs(VOTER_IMAGE_DIR)
    print(f"Created voter image directory: {VOTER_IMAGE_DIR}")

# --- Face Encoding Module Functions ---
def encode_face(image_path, voter_id):
    """
    Detects a single face in an image, extracts its 128-dimensional encoding,
    and associates it with a voter ID.

    Args:
        image_path (str): The path to the image file.
        voter_id (str): The ID of the voter associated with the face.

    Returns:
        tuple: A tuple containing (voter_id, face_encoding) if a single face is found,
               otherwise returns (voter_id, None).
    """
    try:
        image = face_recognition.load_image_file(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return voter_id, None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return voter_id, None

    face_locations = face_recognition.face_locations(image)

    if len(face_locations) == 0:
        print(f"No faces found in {image_path}")
        return voter_id, None
    elif len(face_locations) > 1:
        print(f"Multiple faces found in {image_path}. Expected single face for voter ID {voter_id}")
        return voter_id, None
    else:
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        print(f"Successfully encoded face for voter ID {voter_id} from {image_path}")
        return voter_id, face_encoding

def save_encodings(image_voter_pairs, encodings_file=ENCODINGS_FILE):
    """
    Processes a list of image paths and voter IDs, generates face encodings,
    and saves them to a persistent file.

    Args:
        image_voter_pairs (list): A list of tuples, where each tuple is
                                  (image_path, voter_id).
        encodings_file (str): The file path to save the encodings.
    """
    known_face_encodings = []
    known_face_ids = []

    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
            known_face_encodings = data['encodings']
            known_face_ids = data['ids']
        print(f"Loaded existing encodings from {encodings_file}")

    newly_added_count = 0
    for image_path, voter_id in image_voter_pairs:
        # For dummy data, directly create an encoding to ensure model training can proceed
        # In a real scenario, this would come from face_recognition.face_encodings(image, face_locations)[0]
        if 'dummy' in voter_id: # Check for the dummy voter_id specifically
            # Create a dummy 128-dimensional encoding if it's a dummy voter_id
            # This is important for the initial setup to ensure a model can be trained.
            if voter_id not in known_face_ids:
                encoding = np.zeros(128)
                known_face_encodings.append(encoding)
                known_face_ids.append(voter_id)
                newly_added_count += 1
                print(f"Created and added dummy encoding for voter ID {voter_id}")
            else:
                print(f"Voter ID {voter_id} already exists (dummy). Skipping.")
            continue

        if voter_id in known_face_ids:
            print(f"Voter ID {voter_id} already exists. Skipping {image_path}.")
            continue

        _voter_id, encoding = encode_face(image_path, voter_id)
        if encoding is not None:
            known_face_encodings.append(encoding)
            known_face_ids.append(_voter_id)
            newly_added_count += 1

    # Always create/update the encodings file if there are any encodings or if it needs to be initialized
    if newly_added_count > 0 or not os.path.exists(encodings_file) or (known_face_encodings and known_face_ids):
        with open(encodings_file, 'wb') as f:
            pickle.dump({'encodings': known_face_encodings, 'ids': known_face_ids}, f)
        if newly_added_count > 0:
            print(f"Successfully added {newly_added_count} new voter(s) and saved encodings to {encodings_file}")
        else:
            print(f"Encodings file {encodings_file} initialized/updated (no new voters added). While no new 'real' voters, dummy encodings might have been added.")
    else:
        print("No new voter encodings were added, and file was not initialized.")

# --- KNN Model Training Module Function ---
def train_model(encodings_file, model_file=MODEL_FILE):
    """
    Loads face encodings and voter IDs, trains a KNN classifier, and saves the model.

    Args:
        encodings_file (str): Path to the file containing face encodings and voter IDs.
        model_file (str): Path to save the trained KNN model.
    """
    if not os.path.exists(encodings_file):
        print(f"Error: Encodings file not found at {encodings_file}")
        return

    with open(encodings_file, 'rb') as f:
        data = pickle.load(f)
        known_face_encodings = data['encodings']
        known_face_ids = data['ids']

    if not known_face_encodings:
        print("No face encodings found to train the model.")
        return

    print(f"Loaded {len(known_face_encodings)} face encodings for training.")

    knn_classifier = KNeighborsClassifier(n_neighbors=1, weights='distance')
    knn_classifier.fit(known_face_encodings, known_face_ids)

    with open(model_file, 'wb') as f:
        pickle.dump(knn_classifier, f)

    print(f"Trained KNN model and saved to {model_file}")

# --- Flask API Setup ---
# Global variables to store the loaded KNN model and raw encodings/IDs
current_knn_model = None
known_face_encodings_data = []
known_face_ids_data = []

# Global variables to track last load times
last_model_load_time = 0
last_encodings_load_time = 0

def load_model():
    """
    Loads the trained KNN model and raw face encodings/IDs from respective files,
    only if they are new or updated.
    """
    global current_knn_model, known_face_encodings_data, known_face_ids_data
    global last_model_load_time, last_encodings_load_time

    # --- Load KNN model ---
    if not os.path.exists(MODEL_FILE):
        if current_knn_model is not None:
            print(f"Warning: Model file not found at {MODEL_FILE}. Unloading current model.")
            current_knn_model = None
        last_model_load_time = 0
    else:
        current_model_mtime = os.path.getmtime(MODEL_FILE)
        if current_knn_model is None or current_model_mtime > last_model_load_time:
            try:
                with open(MODEL_FILE, 'rb') as f:
                    current_knn_model = pickle.load(f)
                last_model_load_time = current_model_mtime
                print(f"Successfully loaded/reloaded KNN model from {MODEL_FILE}")
            except Exception as e:
                print(f"Error loading model from {MODEL_FILE}: {e}")
                current_knn_model = None
                last_model_load_time = 0

    # --- Load raw encodings and IDs ---
    if not os.path.exists(ENCODINGS_FILE):
        if known_face_encodings_data or known_face_ids_data:
            print(f"Warning: Encodings file not found at {ENCODINGS_FILE}. Clearing current encodings.")
            known_face_encodings_data = []
            known_face_ids_data = []
        last_encodings_load_time = 0
    else:
        current_encodings_mtime = os.path.getmtime(ENCODINGS_FILE)
        if not known_face_encodings_data or current_encodings_mtime > last_encodings_load_time:
            try:
                with open(ENCODINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    known_face_encodings_data = data.get('encodings', [])
                    known_face_ids_data = data.get('ids', [])
                last_encodings_load_time = current_encodings_mtime
                print(f"Successfully loaded/reloaded {len(known_face_encodings_data)} known face encodings from {ENCODINGS_FILE}")
            except Exception as e:
                print(f"Error loading encodings from {ENCODINGS_FILE}: {e}")
                known_face_encodings_data = []
                known_face_ids_data = []
                last_encodings_load_time = 0


app = Flask(__name__)

# Initial load when the Flask app starts
load_model()

@app.route('/recognize', methods=['POST'])
def recognize_face():
    # Always attempt to load the freshest model and encodings at the start of each request
    load_model()

    global current_knn_model, known_face_encodings_data, known_face_ids_data

    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected image file'}), 400

    try:
        # Read the image file content into a BytesIO object
        image_stream = BytesIO(file.read())
        # Load image using face_recognition (Pillow will handle it)
        image = face_recognition.load_image_file(image_stream)

        face_locations = face_recognition.face_locations(image)

        if len(face_locations) == 0:
            return jsonify({'status': 'success', 'voter_id': 'No face found'}), 200
        elif len(face_locations) > 1:
            return jsonify({'status': 'success', 'voter_id': 'Multiple faces found'}), 200
        else:
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]

            # Ensure model and encodings are loaded after `load_model()` call
            if current_knn_model is None or not known_face_encodings_data:
                return jsonify({'status': 'error', 'message': 'Recognition model or encodings not loaded or found.'}), 500

            # Predict the voter ID using the KNN model
            # First, find the closest known face encoding
            distances = face_recognition.face_distance(known_face_encodings_data, face_encoding)
            min_distance_idx = np.argmin(distances)

            # A common threshold for face recognition (lower is more strict)
            # You might need to tune this threshold based on your dataset
            if distances[min_distance_idx] < 0.6:
                # If close enough, use the KNN classifier to get the predicted ID
                # The KNN model was trained on these encodings and IDs, so its prediction
                # for the closest encoding should correspond to the correct ID.
                predicted_id = current_knn_model.predict([face_encoding])[0]
                return jsonify({'status': 'success', 'voter_id': predicted_id}), 200
            else:
                return jsonify({'status': 'success', 'voter_id': 'Unknown'}), 200

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing image: {e}'}), 500


if __name__ == '__main__':
    # Use 0.0.0.0 to make the server accessible from outside the container
    # This block will not execute in a typical Colab environment where __name__ != '__main__'
    # unless run directly. It's primarily for local testing.
    print("Flask app setup complete. To run, please execute `app.run(host='0.0.0.0', port=5000)` in a non-blocking manner or via a service like ngrok.")
    print("For example, after setting up ngrok, you can call `app.run(host='0.0.0.0', port=5000)` in a separate thread or use `!python -c 'from your_script import app; app.run(host=\"0.0.0.0\", port=5000)' &` for background execution in a terminal.")
    # app.run(host='0.0.0.0', port=5000) # Uncomment to run if in a suitable environment
