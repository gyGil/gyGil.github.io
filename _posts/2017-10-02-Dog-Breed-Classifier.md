---
title: "Dog Breed Classifier"
excerpt: ""
last_modified_at:
categories:
  - Visual Recognition
tags:
  - CNN
  - Transfer Learning
  - Keras
  - TensorFlow
---

## Introduction
This project is for identifying canine breed given an image of a dog. If supplied an image of a human, the program will identify the resembling dog breed.
I implemented CNN using not transfer learning and CNNs using transfer learning with VGG-16, VGG-19 to classify dog breed. At the end of this post, I will compare 3 models for performance.

## Process
* Step 0: Import Datasets
* Step 1: Detect Humans
* Step 2: Detect Dogs with ResNet-50
* Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
* Step 4: Create a CNN to Classify Dog Breeds with VGG-16 (using Transfer Learning)
* Step 5: Create a CNN to Classify Dog Breeds with VGG-19 (using Transfer Learning)
* Step 6: Compare above 3 models

## Environment
* AWS EC2 p2.xlarge
* Jupyter Notebook
* Python 3.5, Keras, TensorFlow 1.1

## Step 0: Import Datasets
Import Dog dataset and divide into train, validation, and test datasets. Dog images is composed of 8,351 images which are categorized into 133 breeds.
```python
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('dogImages/train')
valid_files, valid_targets = load_dataset('dogImages/valid')
test_files, test_targets = load_dataset('dogImages/test')
```

Import 13,233 human images.  
```python
import random
random.seed(8675309)

# load filenames in shuffled human dataset
human_files = np.array(glob("lfw/*/*"))
random.shuffle(human_files)
```

## Step 1: Detect Humans
I used [Haar feature-based cascade classifiers](https://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html) to detect human faces in images. OpenCV provides many pre-trained face detectors, stored as XML files on [github](https://github.com/opencv/opencv/tree/master/data/haarcascades).

```python
import cv2    

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0
```

## Step 2: Detect Dogs with ResNet-50
I used a pre-trained ResNet-50 model to detect dogs in images.
```python
from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')
```

### Pre-process the data
When using TensorFlow as backend, Keras CNNs require a 4D array as input, with shape
**(nb_samples,rows,columns,channels)**  
```python
from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
```

### Dog detector
It makes the prediction using resnet50. If a image is classified between 151 to 268 using ResNet50, it is a dog image.

```python
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image                  
from tqdm import tqdm

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))
```

## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
I will build a simple 3 layers of CNN with a dense layer from beginning without Transfer Learning for test purpose. Building CNN from the start requires a lot of dataset and training time. It is demonstration to show the weakness of building CNN from scratch compare to Transfer Learning.

### pre-process the data.
```python
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255
```

### Build model.

Layer (type)                           | Output Shape             | Param #   
---------------------------------------|--------------------------|-----------
conv2d_4 (Conv2D)                      | (None, 224, 224, 16)     |  208          
max_pooling2d_6 (MaxPooling2           | (None, 112, 112, 16)     |  0       
conv2d_5 (Conv2D)                      | (None, 112, 112, 32)     |  2080      
max_pooling2d_7 (MaxPooling2           | (None, 56, 56, 32)       |  0   
conv2d_6 (Conv2D)                      | (None, 56, 56, 64)       |  8256  
max_pooling2d_8 (MaxPooling2           | (None, 28, 28, 64)       |  0
global_average_pooling2d_4 (           | (None, 64)               |  0
dense_5 (Dense)                        | (None, 133)              |  8645         


Total params: 19,189  
Trainable params: 19,189  
Non-trainable params: 0  

```python
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding='same',
                        activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=2))

# https://keras.io/layers/pooling/
model.add(GlobalAveragePooling2D())
model.add(Dense(133,activation='softmax'))
```

### Train models
```python
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint  

epochs = 20

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
```

## Step 4: Create a CNN to Classify Dog Breeds with VGG-16 (using Transfer Learning)
## Step 5: Create a CNN to Classify Dog Breeds with VGG-19 (using Transfer Learning)
## Step 6: Compare above 3 models
## Reference
[1] Artificial Intelligence. (n.d.). Retrieved from https://www.udacity.com/course/ai-artificial-intelligence-nanodegree--nd898
