# scripts/train_pneumonia.py

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Define Paths and Hyperparameters
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'val')
TEST_DIR = os.path.join(BASE_DIR, 'dataset', 'test')

IMG_SIZE = 150 # Standardize all X-rays to 150x150 pixels
BATCH_SIZE = 32
EPOCHS = 15

print("[INFO] Preparing Data Generators...")

# 2. Data Augmentation (Careful with medical images: no crazy flips!)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Validation and testing only need rescaling (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary' # Because we only have 2 classes: NORMAL vs PNEUMONIA
)

val_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# 3. Build the Custom CNN Architecture
print("[INFO] Building Custom CNN...")
model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Prevents overfitting
    Dense(1, activation='sigmoid') # Sigmoid is perfect for binary classification
])

# 4. Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 5. Train the Model with Callbacks
print("[INFO] Training Model...")

# Ensure models directory exists
model_dir = os.path.join(BASE_DIR, "models")
os.makedirs(model_dir, exist_ok=True)
model_save_path = os.path.join(model_dir, "pneumonia_detector.h5")

# Save the best model automatically during training
checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=max(1, val_generator.samples // BATCH_SIZE),
    callbacks=[checkpoint]
)

print("[INFO] Model Training Complete and Saved!")