---
title: "Facial Keypoint Detection"
excerpt: ""
last_modified_at:
categories:
  - Visual Recognition
tags:
  - CNN
  - Keras
  - TensorFlow
  - OpenCV
---

## Introduction
This project is to find the important keypoints on a face by using OpenCV and CNN. After finding the keypoints, the program generate the image and camera stream to wear sunglasses using keypoints on the faces. At the end of this blog, I will compare the performance among the various deep learning models.

## Process
* Step 0: Import datasets
* Step 1: Build CNN model to find keypoints on face
* Step 2: Preprocess image
* Step 3: Face Detection with OpenCV
* Step 4: Final product
* Step 5: Compare various CNN models

## Environment
* Laptop with CUDA
* Jupyter Notebook
* Python 3.5, Keras 2.2, TensorFlow-gpu 1.8

## Step 0: Import Datasets
I used dataset from Kaggle[Kaggle](https://www.kaggle.com/c/facial-keypoints-detection/data). The dataset is composed of 2140 train set and 1784 test set. The csv data point file contains 15 keypoints(x, y values) and 96x96 image gray color values(0-255) with 1 channel. And each image is a 96x96 face image with 1 channel (gray). The dataset is normalized to make better training. Color value is normalized between 0.0 and 1.0 and x, y values for key points on face are normalized between -1.0 and 1.0.

```python
from utils import *

# Load training set
X_train, y_train = load_data()
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# Load testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))
```
X_train.shape == (2140, 96, 96, 1)  
y_train.shape == (2140, 30); y_train.min == -0.920; y_train.max == 0.996  
X_test.shape == (1783, 96, 96, 1)  

### Visualize the training data

```python
import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure(figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_data(X_train[i], y_train[i], ax)
```
![train data](/images/fkd_train_data.png)

## Step 1: Build CNN model to find keypoints on face
I tried to test various models to get the better result. I tested single and multiple conv + maxpool layers with 1~3 fully connected layers at the end of model. I also tested dropout to generalize on conv layer which isn't work because conv layers is already generalized itself (conv layers's local connectivity). Dropout is applied to fully connected layer to reduce overfitting. I used ReLu activations through model except the final activation. I chose final activation as Tanh because final output is -1 to 1. The adaptive optimizer which is Adam was produced the best result. Loss function is applied mean squared error that can evaluate continuous numerical outputs. Early Stopping is used to stop when validation loss is not improved in few epochs. The below final model is inspired from VGG-16 and VGG-19.

```python
# Import deep learning resources from Keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense

model = Sequential()

model.add(Convolution2D(filters=16, kernel_size=3, padding='same', activation='relu',
                       input_shape=(96, 96, 1)))
model.add(Convolution2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(Convolution2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Convolution2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='tanh'))

# Summarize the model
model.summary()
```

Layer (type)                           | Output Shape             | Param #   
---------------------------------------|--------------------------|-----------
conv2d_11 (Conv2D)                     | (None, 96, 96, 16)       |  160          
conv2d_12 (Conv2D)                     | (None, 96, 96, 16)       |  2320       
max_pooling2d_5 (MaxPooling2           | (None, 48, 48, 16)       |  0      
conv2d_13 (Conv2D)                     | (None, 48, 48, 32)       |  4640   
conv2d_14 (Conv2D)                     | (None, 48, 48, 32)       |  9248  
max_pooling2d_6 (MaxPooling2           | (None, 24, 24, 32)       |  0
conv2d_15 (Conv2D)                     | (None, 24, 24, 64)       |  18496
conv2d_16 (Conv2D)                     | (None, 24, 24, 64)       |  36928
conv2d_17 (Conv2D)                     | (None, 24, 24, 64)       |  36928  
max_pooling2d_7 (MaxPooling2           | (None, 24, 24, 64)       |  0
conv2d_18 (Conv2D)                     | (None, 12, 12, 128)      |  73856
conv2d_19 (Conv2D)                     | (None, 12, 12, 128)      |  147584
conv2d_20 (Conv2D)                     | (None, 12, 12, 128)      |  147584
max_pooling2d_8 (MaxPooling2           | (None, 6, 6, 128)        |  0
flatten_2 (Flatten)                    | (None, 4608)             |  0
dense_3 (Dense)                        | (None, 512)              |  2359808
dropout_2 (Dropout)                    | (None, 512)              |  0
dense_4 (Dense)                        | (None, 30)               |  15390


Total params: 2,852,942  
Trainable params: 2,852,942  
Non-trainable params: 0  

### Training and loss/ accuracy
```python
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint

adam = Adam() #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) default epsilon = 1e-07
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae', 'acc'])

earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath='best_my_model.hdf5')
callbacks_list = [earlystop, checkpointer]

hist = model.fit(X_train, y_train, batch_size=16, epochs=3000, verbose=2,  callbacks=callbacks_list, validation_split=0.2, shuffle=True)

model.save('my_model.h5')
```

- 145s - loss: 0.0010 - mean_absolute_error: 0.0239 - acc: 0.8002 - val_loss: 9.6063e-04 - val_mean_absolute_error: 0.0219 - val_acc: 0.8037  
Epoch 45/3000  
- 140s - loss: 0.0010 - mean_absolute_error: 0.0237 - acc: 0.8020 - val_loss: 9.4013e-04 - val_mean_absolute_error: 0.0217 - val_acc: 0.7827  
Epoch 46/3000  
- 145s - loss: 0.0010 - mean_absolute_error: 0.0235 - acc: 0.7967 - val_loss: 9.1961e-04 - val_mean_absolute_error: 0.0214 - val_acc: 0.7944  
Epoch 47/3000  
- 147s - loss: 0.0010 - mean_absolute_error: 0.0237 - acc: 0.8072 - val_loss: 9.0973e-04 - val_mean_absolute_error: 0.0213 - val_acc: 0.7874  
Epoch 48/3000  
- 148s - loss: 9.7474e-04 - mean_absolute_error: 0.0231 - acc: 0.8078 - val_loss: 9.3755e-04 - val_mean_absolute_error: 0.0216 - val_acc: 0.8084  
Epoch 49/3000  
- 149s - loss: 9.7831e-04 - mean_absolute_error: 0.0232 - acc: 0.7956 - val_loss: 8.6974e-04 - val_mean_absolute_error: 0.0210 - val_acc: 0.8154  
Epoch 00049: early stopping  

## Step 2: Preprocess image


## Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
I will build a simple 3 layers of CNN with a dense layer from beginning without Transfer Learning for comparison purpose. Building CNN from the start requires a lot of dataset and training time. It is demonstration to show the weakness of building CNN from scratch compare to Transfer Learning.

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

## Step 2: Preprocess image

## Step 4: Create a CNN to Classify Dog Breeds with VGG-16 (using Transfer Learning)
I used Transfer Learning with VGG-16 and a dense layer to train model faster and get better accuracy.

Layer (type)                           | Output Shape             | Param #   
---------------------------------------|--------------------------|-----------
global_average_pooling2d_5             | (None, 512)              |  0          
dense_6 (Dense)                        | (None, 133)              |  68229       

Total params: 68,229  
Trainable params: 68,229  
Non-trainable params: 0  

### Train the model
```python
bottleneck_features = np.load('bottleneck_features/DogVGG16Data.npz')
train_VGG16 = bottleneck_features['train']
valid_VGG16 = bottleneck_features['valid']
test_VGG16 = bottleneck_features['test']

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(133, activation='softmax'))

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.VGG16.hdf5',
                               verbose=1, save_best_only=True)

# Train model
VGG16_model.fit(train_VGG16, train_targets,
          validation_data=(valid_VGG16, valid_targets),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)
```
Epoch 20/20 loss: 8.2550 - acc: 0.4740 - val_loss: 8.6555 - val_acc: 0.4012  

### Test the model
```python
VGG16_model.load_weights('saved_models/weights.best.VGG16.hdf5')

# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100*np.sum(np.array(VGG16_predictions)==np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```
Test accuracy: 39.9522%  

## Step 5: Create a CNN to Classify Dog Breeds with VGG-19 (using Transfer Learning)
I used Transfer Learning with VGG-19 and 2 dense layers to get better accuracy than above models.

Layer (type)                           | Output Shape             | Param #   
---------------------------------------|--------------------------|-----------
global_average_pooling2d_6             | (None, 512)              |  0          
dense_7 (Dense)                        | (None, 50)               |  25650        
dense_8 (Dense)                        | (None, 133)              |  6783  

Total params: 32,433  
Trainable params: 32,433  
Non-trainable params: 0  

### Train the model
```python
bottleneck_features = np.load('bottleneck_features/DogVGG19Data.npz')
train_VGG19 = bottleneck_features['train']
valid_VGG19 = bottleneck_features['valid']
test_VGG19 = bottleneck_features['test']

VGG19_model = Sequential()
VGG19_model.add(GlobalAveragePooling2D(input_shape = train_VGG19.shape[1:]))
VGG19_model.add(Dense(50))
VGG19_model.add(Dense(133, activation='softmax'))

# Train model
checkpointer = ModelCheckpoint(filepath='saved_models/weights.bet.VGG19.hdf5',
                              verbose = 1, save_best_only=True)

VGG19_model.fit(train_VGG19, train_targets,
               validation_data=(valid_VGG19, valid_targets),
               epochs=40, batch_size=20, callbacks=[checkpointer], verbose = 1)
```
Epoch 20/40 loss: 0.0616 - acc: 0.9832 - val_loss: 1.6689 - val_acc: 0.7257     

### Test the model
```python
VGG19_model.load_weights('saved_models/weights.bet.VGG19.hdf5')

VGG19_predictions = [np.argmax(VGG19_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG19]

test_accuracy = 100 * np.sum(np.array(VGG19_predictions)==np.argmax(test_targets, axis=1))/len(VGG19_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)
```
Test accuracy: 72.2488%  

## Step 6: Compare above 3 models
The condition of training for all model:  
20 epochs, learning rate  

Structure      |    # Dense Layer   | Transfer Learning  | Test Accuracy               
---------------|--------------------|--------------------|---------------
CNN: 3 Layers  |  1 (133 units)     |          X         | 6.10%                      
CNN: VGG-16    |  1 (133 units)     |          O         | 39.95%              
CNN: VGG-19    |  2 (50, 133 units) |          O         | 72.24%                 

As you see, we can improve accuracy and reduce the training time a lot with Transfer Learning. Also VGG-19 has the better result than VGG-16 with 2 dense layers. VGG-19 has more stacked layers with more filters. It means it has more representation power than VGG-16 normally. Which means it can have better accuracy too. However, when we train more stacked CNN from scratch, it spends more memory, computation power, and time.  
Also, we can figure out that we put 2 dense layers at the end of model, it leads to better accuracy than 1 dense layer.  

## Outcome

```python
dog_human_detector("images/man-852762_960_720.jpg")
dog_human_detector("images/GettyImages-694355292-1503877610-640x426.jpg")
dog_human_detector("images/Taka_Shiba.jpg")
dog_human_detector("images/eebabf5825e8247d99ac2cd118db840ff31d7bfa_hq.jpg")
dog_human_detector("images/hotdog-taco-dog-today-161029-tease_845d920c7ea63371a9bf48203d22036f.jpg")
dog_human_detector("images/dog-human-hybrid-woman.jpeg")
```

Hello, human!  
![human 1](/images/dog1.png)
You look like Cairn_terrier  

Hello, human!  
![human 2](/images/dog2.png)
You look like German_shepherd_dog  

Hello, human!  
![dog 1](/images/dog3.png)
You look like Akita  

Hi, Dog!    
![dog 2](/images/dog4.png)
You look like Bull_terrier  

Hi, Dog!  
![dog 3](/images/dog5.png)
You look like Dachshund  

Hi, Dog!  
![dog 4](/images/dog6.png)
You look like English_toy_spaniel  



[Source Code](https://github.com/gyGil/Dog-Breed-Classifier)  

## Reference
[1] Facial Keypoints Detection | Kaggle. (n.d.). Retrieved from https://www.kaggle.com/c/facial-keypoints-detection/data
