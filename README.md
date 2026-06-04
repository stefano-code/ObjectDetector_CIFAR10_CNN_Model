# CIFAR-10 Image Classification with CNN (Keras / TensorFlow)

This project implements a **Convolutional Neural Network (CNN)** using Keras (TensorFlow backend) to classify images from the **CIFAR-10 dataset**. The model is trained to recognize 10 different object categories such as airplanes, cars, birds, cats, and more.

---

## 📌 Project Overview

The goal of this project is to build and train a deep learning model capable of performing image classification on small RGB images (32x32 pixels).

The workflow includes:
- Loading and exploring the CIFAR-10 dataset
- Preprocessing images and labels
- Building a CNN architecture
- Training the model
- Evaluating performance on test data

---

## 📊 Dataset: CIFAR-10

The CIFAR-10 dataset consists of:
- **60,000 images** (32x32 RGB)
- **10 classes**
- **50,000 training images**
- **10,000 test images**

### Classes:
- Airplane ✈️  
- Automobile 🚗  
- Bird 🐦  
- Cat 🐱  
- Deer 🦌  
- Dog 🐶  
- Frog 🐸  
- Horse 🐴  
- Ship 🚢  
- Truck 🚚  

---

## 🧹 Data Preprocessing

Before training, the dataset is processed as follows:

- Images are converted to `float32`
- Pixel values are normalized to the range `[0, 1]`
- Labels are converted to **one-hot encoding** using `to_categorical`

```python
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
```

---

## 🧠 Model Architecture

The model is a deep Convolutional Neural Network built using Sequential API.

Structure:
- Multiple Conv2D layers with ReLU activation
- MaxPooling2D layers for downsampling
- Dropout layers to reduce overfitting
- Fully connected Dense layers
- Final Softmax layer for classification

### Simplified architecture:
- Conv2D → ReLU → Conv2D → ReLU → MaxPooling → Dropout
- Conv2D → ReLU → Conv2D → ReLU → MaxPooling → Dropout
- Conv2D → ReLU → Conv2D → ReLU → MaxPooling → Dropout
- Flatten → Dense(512) → Dropout → Dense(128) → Dense(10)
---

## ⚙️ Compilation Settings

The model is compiled using:

- Loss function: Categorical Crossentropy
- Optimizer: Adam
- Metrics: Accuracy
```python 
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---
## 🚀 Training

The model is trained using:

- Batch size: 256
- Epochs: 20
- Validation data: CIFAR-10 test set
```python 
history = model.fit(
    x_train, y_train,
    batch_size=256,
    epochs=20,
    validation_data=(x_test, y_test)
)
```

---
## 📈 Evaluation
After training, the model is evaluated on the test dataset:
```python 
score = model.evaluate(x_test, y_test)
```
This returns:
- Test loss
- Test accuracy

---
## 🖼️ Data Visualization
Some sample images from the dataset are displayed to understand the input data distribution.

---
## 📦 Requirements
To run this project, install the required libraries:
```python 
pip install tensorflow keras numpy matplotlib
```
---

## ▶️ How to Run
1. Clone the repository:
```python
git clone https://github.com/stefano-code/cifar10-cnn.git
cd cifar10-cnn
```
2. Open the notebook:
```python
jupyter notebook
```
3. Run all cells sequentially.

---

## 📌 Notes
- This model is a baseline CNN for CIFAR-10 classification.
- Performance can be improved using:
    - Data augmentation
    - Batch normalization
    - Deeper architectures (ResNet, VGG, etc.)

---
## 📜 License
This project is released under MIT License.

---


