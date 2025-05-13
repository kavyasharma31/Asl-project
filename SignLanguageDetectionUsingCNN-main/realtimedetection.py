from keras.models import model_from_json
import cv2
import numpy as np
import random

# Load model from JSON and weights
def load_model(json_path, weights_path):
    with open(json_path, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    print("Model loaded successfully from", json_path, "and", weights_path)
    return model

# Feature extraction function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 400, 400, 1)  # Reshape to (1, 400, 400, 1)
    return feature / 255.0  # Normalize to [0, 1]

# Epsilon-greedy prediction function
def epsilon_greedy_prediction(model, features, label, epsilon=0.1):
    if random.random() < epsilon:
        # Exploration: Randomly pick a label
        random_label = random.choice(label)
        random_confidence = random.uniform(0, 100)  # Random confidence
        print(f"Exploration: Random label '{random_label}' chosen.")
        return random_label, random_confidence
    else:
        # Exploitation: Use the model's prediction
        pred = model.predict(features)
        prediction_label = label[pred.argmax()]  # Get the predicted label
        confidence = np.max(pred) * 100  # Get accuracy of prediction
        print(f"Exploitation: Model predicted '{prediction_label}' with {confidence:.2f}% confidence.")
        return prediction_label, confidence

# Load the model
model = load_model("signlanguagedetectionmodel400x400.json", "signlanguagedetectionmodel400x400.h5")

# Label mapping for predictions
label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
         'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 
         'u', 'v', 'w', 'x', 'y', 'z']

# Start video capture
cap = cv2.VideoCapture(0)

epsilon = 0.1  # Set epsilon for exploration probability

while True:
    ret, frame = cap.read()  # Read the frame from the webcam
    if not ret:
        print("Failed to grab frame")
        break
    
    # Draw a rectangle to define the region of interest (ROI)
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    
    # Crop the region of interest
    cropframe = frame[40:300, 0:300]
    
    # Preprocess the image (convert to grayscale and resize to 400x400)
    cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
    cropframe = cv2.resize(cropframe, (400, 400))
    
    # Extract features from the preprocessed image
    cropframe = extract_features(cropframe)
    
    # Apply epsilon-greedy to get the prediction or random output
    prediction_label, accu = epsilon_greedy_prediction(model, cropframe, label, epsilon)
    
    # Display prediction on the frame
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 165, 255), -1)
    cv2.putText(frame, f'{prediction_label}  {accu:.2f}%', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show the frame with the prediction
    cv2.imshow("Sign Language Detection", frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
