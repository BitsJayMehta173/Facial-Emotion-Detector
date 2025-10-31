# Emotion Detection from Images

## Project Overview
This project aims to develop a deep learning model for detecting and classifying emotions from facial images. The model is trained on the **FER-2013** dataset and Tested on Python Webcam application.

## Custom Model Details
The model is a **Custom Convolutional Neural Network (CNN)** designed specifically for emotion classification. The architecture consists of:
- **Four convolutional layers** with Batch Normalization and LeakyReLU activation.
- **MaxPooling layers** for downsampling.
- **Fully connected layers** with Dropout for regularization.
- **Final softmax layer** to classify images into 7 emotion categories.

This model was chosen for its balance between complexity and efficiency, achieving an optimal accuracy while maintaining reasonable training time.

## Technologies Used
- **Python**
- **PyTorch**
- **OpenCV**
- **FER-2013 Dataset**

## Installation
```bash
pip install -r requirements.txt
```

## Training the Model
To train the custom model, run using Python Build Environment:
Open cust_model_training.ipynb 
then Try running cells
The machine must be installed with Python and Jupyter extensions to run all the cells
Install if not installed
Then Wait for all the cells to Complete Running

## Running the Facial Emotion Detection App
To launch application:
```bash
python emotion_detection_app.py
```

## Future Improvements
- Experiment with different CNN architectures.
- Fine-tune hyperparameters for better accuracy.
- Integrate more datasets for improved generalization.

## Contributors
- Jay Mehta (M25CSE034)
- Aryan Barnawal (M25CSE035)
- Kshitiz Budhathoki (M25CSE036)


# emotion-detection

