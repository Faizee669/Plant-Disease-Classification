<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease Classification using Deep Learning</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #333;
        }
        pre {
            background: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
        }
        a {
            color: #1a0dab;
        }
    </style>
</head>
<body>
    <h1>Plant Disease Classification using Deep Learning</h1>
    <p>This repository contains the code and resources for my Final Year Project, which involves building a deep learning model to classify plant diseases from images. The project utilizes transfer learning with a pre-trained DenseNet169 model and fine-tunes it to accurately classify different types of plant diseases.</p>
    
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#introduction">Introduction</a></li>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#model-architecture">Model Architecture</a></li>
        <li><a href="#training">Training</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
        <li><a href="#results">Results</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#license">License</a></li>
    </ul>
    
    <h2 id="introduction">Introduction</h2>
    <p>Plant diseases are a major threat to food security and agricultural productivity. Early detection and accurate classification of these diseases can help in taking timely actions to prevent their spread. This project aims to develop a deep learning model capable of classifying plant diseases from images using transfer learning with DenseNet169.</p>
    
    <h2 id="dataset">Dataset</h2>
    <p>The dataset used for this project is the <a href="https://www.kaggle.com/emmarex/plantdisease" target="_blank">PlantVillage dataset</a>. It contains images of healthy and diseased plant leaves categorized into different classes. For this project, we focus on the following classes:</p>
    <ul>
        <li>Potato___Early_blight</li>
        <li>Potato___Late_blight</li>
        <li>Potato___healthy</li>
    </ul>
    
    <h2 id="installation">Installation</h2>
    <p>To get started with the project, clone this repository and install the required dependencies:</p>
    <pre><code>git clone https://github.com/yourusername/plant-disease-classification.git
cd plant-disease-classification
pip install -r requirements.txt</code></pre>
    
    <h2 id="model-architecture">Model Architecture</h2>
    <p>The model architecture is based on the DenseNet169 network. The pre-trained DenseNet169 model is used as the base model, and additional layers are added for fine-tuning.</p>
    <pre><code># Model Initialization
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

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])</code></pre>
    
    <h2 id="training">Training</h2>
    <p>The training data is augmented using <code>ImageDataGenerator</code> to improve the model's robustness and generalization. The model is trained for 50 epochs.</p>
    <pre><code>model_history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=50,
    verbose=1
)</code></pre>
    
    <h2 id="evaluation">Evaluation</h2>
    <p>The trained model is evaluated on a separate test dataset to measure its performance.</p>
    <pre><code>test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")</code></pre>
    
    <h2 id="results">Results</h2>
    <p>The model's performance is visualized through training and validation loss and accuracy plots.</p>
    <pre><code># Plotting Training and Validation Loss
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
plt.show()</code></pre>
    
    <h2 id="usage">Usage</h2>
    <p>To use the model for predicting plant diseases on new images, use the following code:</p>
    <pre><code>from tensorflow.keras.preprocessing.image import load_img, img_to_array

img_path = 'path/to/your/image.jpg'
img = load_img(img_path, target_size=(256, 256))
img = img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]
print(f"The image is predicted as {predicted_class}")</code></pre>
    
    <h2 id="contributing">Contributing</h2>
    <p>Contributions are welcome! Please feel free to submit a Pull Request.</p>
    
    <h2 id="license">License</h2>
    <p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for more details.</p>
</body>
</html>
