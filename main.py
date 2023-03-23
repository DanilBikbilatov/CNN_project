# Import necessary libraries
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# Define the paths to the image and watermark directories
image_dir = '/Users/biktirus/Desktop/WM_hm/dataset/orig1'
watermark_dir = '/Users/biktirus/Desktop/WM_hm/dataset/wat1'

# Load the images and watermarks into separate arrays
images = []
watermarks = []

for filename in os.listdir(image_dir):
    img = cv2.imread(os.path.join(image_dir, filename))
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    images.append(img)
    
    watermark = cv2.imread(os.path.join(watermark_dir, filename), cv2.IMREAD_GRAYSCALE)
    watermark = cv2.resize(watermark, (128, 128))
    watermark = watermark.astype('float32') / 255.0
    watermarks.append(watermark)

# Convert the image and watermark arrays to NumPy arrays
images = np.array(images)
watermarks = np.array(watermarks)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, watermarks, test_size=0.2, random_state=42)

# Define the embedding module
def embedding_module(input_shape):
    # Define the model
    model = Sequential()
    # Add a convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation function
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # Add a max pooling layer with a 2x2 pool size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add a flattening layer to convert the 2D feature map to a 1D feature vector
    model.add(Flatten())
    # Add a fully connected layer with 64 neurons and ReLU activation function
    model.add(Dense(64, activation='relu'))
    # Add a final fully connected layer with sigmoid activation function to output the embedded image with watermark
    model.add(Dense(input_shape[0]*input_shape[1], activation='sigmoid'))

    return model

# Define the extraction module
def extraction_module(input_shape):
    # Define the model
    model = Sequential()
    # Add a convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation function
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    # Add a max pooling layer with a 2x2 pool size
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Add a flattening layer to convert the 2D feature map to a 1D feature vector
    model.add(Flatten())
    # Add a fully connected layer with 64 neurons and ReLU activation function
    model.add(Dense(64, activation='relu'))
    # Add a final fully connected layer with sigmoid activation function to output the extracted watermark
    model.add(Dense(input_shape[0]*input_shape[1], activation='sigmoid'))

    return model

# Define the input shape
input_shape = (128, 128, 1)

# Define the embedding module
embedding_model = embedding_module(input_shape)
# Compile the model with binary crossentropy loss and Adam optimizer
embedding_model.compile(loss='binary_crossentropy', optimizer='adam')

# Define the extraction module
extraction_model = extraction_module(input_shape)
# Compile the model with binary crossentropy loss and Adam optimizer
extraction_model.compile(loss='binary_crossentropy', optimizer='adam')

# Train the embedding module
embedding_model.fit(x_train, y_train, epochs=10, batch_size=32)

# Train the extraction module
extraction_model.fit(x_train, y_train, epochs=10, batch_size=32)

# Test the embedding module
embedded_image = embedding_model.predict(np.array([image]), np.array([watermark]))

# Test the extraction module
extracted_watermark = extraction_model.predict(embedded_image)