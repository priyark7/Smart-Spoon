
import tensorflow as tf
from tensorflow.keras import layers, models

# Simple CNN for food image classification
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # e.g., [low salt, medium, high]
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("CNN model ready. Train using model.fit() with your dataset.")
