# scripts/predict_xray.py

import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. Define Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pneumonia_detector.h5')

# Ensure the model exists before trying to load it
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at {MODEL_PATH}")
    print("Please run 'python scripts/train_pneumonia.py' to train and save the model first.")
    sys.exit()

print("[INFO] Loading trained Pneumonia Detection model...")
model = load_model(MODEL_PATH)

# 2. Define the prediction function
def predict_pneumonia(img_path):
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found at {img_path}")
        return

    # Preprocess the image exactly how we trained the model
    # Resize to 150x150 pixels
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    
    # Keras expects a "batch" of images, so we add an extra dimension (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale the pixel values to between 0 and 1 (crucial step!)
    img_array /= 255.0

    # 3. Run the prediction
    # The sigmoid activation function returns a value between 0 (Normal) and 1 (Pneumonia)
    prediction = model.predict(img_array)[0][0]
    
    print("\n" + "="*40)
    if prediction >= 0.5:
        print(f"🩺 DIAGNOSIS: PNEUMONIA DETECTED")
        print(f"📊 Confidence: {prediction * 100:.2f}%")
    else:
        print(f"🩺 DIAGNOSIS: NORMAL (Clear Lungs)")
        print(f"📊 Confidence: {(1 - prediction) * 100:.2f}%")
    print("="*40 + "\n")

# 4. User Interaction
# Allows the user to run the script and pass an image path directly via the terminal
if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
        predict_pneumonia(test_image_path)
    else:
        print("Usage: python scripts/predict_xray.py <path_to_image>")
        # Fallback interactive prompt if no argument is provided
        user_input = input("Enter the file path of the X-ray image to test: ").strip()
        # Remove quotes if the user drags and drops the file into the terminal
        user_input = user_input.replace('"', '').replace("'", "")
        predict_pneumonia(user_input)