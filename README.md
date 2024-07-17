# Plant Disease Classification using Deep Learning
This repository contains the code and resources for my Final Year Project, which involves building a deep learning model to classify plant diseases from images. The project utilizes transfer learning with a pre-trained DenseNet169 model and fine-tunes it to accurately classify different types of plant diseases.
# Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)

## Introduction
Plant diseases are a major threat to food security and agricultural productivity. Early detection and accurate classification of these diseases can help in taking timely actions to prevent their spread. This project aims to develop a deep learning model capable of classifying plant diseases from images using transfer learning with DenseNet169.

## Dataset
The dataset used for this project is the PlantVillage dataset. It contains images of healthy and diseased plant leaves categorized into different classes. For this project, we focus on the following classes:

Potato___Early_blight
Potato___Late_blight
Potato___healthy
## Installation
To get started with the project, clone this repository and install the required dependencies:
```bash
git clone https://github.com/Faizee669/Plant-Disease-Classification
cd plant-disease-classification
pip install -r requirements.txt
```
## Model Architecture
The model architecture is based on the DenseNet169 network. The pre-trained DenseNet169 model is used as the base model, and additional layers are added for fine-tuning.
```bash
# Model Initialization
base_model = DenseNet169(input_shape=(256, 256, 3), include_top=False, weights="imagenet")

# Freezing Layers
for layer in base_model.layers:
    layer.trainable = False

# Building Model
model = Sequential([
    base_model,
    Dropout(0.5),
    Flatten(),
    BatchNormalization(),
    Dense(2048, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(1024, kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```
## Training
The training data is augmented using ImageDataGenerator to improve the model's robustness and generalization. The model is trained for 50 epochs.
```bash
model_history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=50,
    verbose=1
)
```
## Evaluation
The trained model is evaluated on a separate test dataset to measure its performance.
```bash
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
```
## Results
The model's performance is visualized through training and validation loss and accuracy plots.
```bash
# Plotting Training and Validation Loss
plt.plot(model_history.history['loss'], label='Train Loss')
plt.plot(model_history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting Training and Validation Accuracy
plt.plot(model_history.history['accuracy'], label='Train Accuracy')
plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
## Usage
To use the model for predicting plant diseases on new images, use the following code:
```bash
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_path = 'path/to/your/image.jpg'
img = load_img(img_path, target_size=(256, 256))
img = img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]
print(f"The image is predicted as {predicted_class}")
```

