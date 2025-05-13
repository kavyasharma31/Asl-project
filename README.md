#🤟 American Sign Language (ASL) Recognition using Deep Learning
This project focuses on building a deep learning model capable of recognizing American Sign Language (ASL) hand gestures from images. The model classifies static hand signs representing the 26 letters (A-Z) and digits (0-9) using a custom dataset.

📌 Features
Classifies 36 ASL characters (A–Z, 0–9)

Convolutional Neural Network (CNN) based architecture

Image preprocessing and augmentation for improved performance

High accuracy with low training time

Real-time prediction capability (optional extension with webcam)

#📁 Dataset
The dataset consists of folders named 0–9 and a–z, each containing JPEG images of corresponding hand signs.

#⚙️ Technologies Used
Python

TensorFlow / Keras

OpenCV (for optional real-time demo)

NumPy, Pandas

Matplotlib, Seaborn (for visualization)

#🧠 Model Architecture
Input Layer: Resized grayscale images (e.g., 64x64)

Conv2D + MaxPooling layers

Dropout layers for regularization

Flatten + Dense layers

Output Layer: Softmax with 36 units (A–Z, 0–9)

