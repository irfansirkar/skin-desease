# Skin Cancer Detection Using Deep Learning

A CNN-based skin cancer detection system with multiple interfaces for real-time prediction.

## Features

- **Deep Learning Model**: CNN architecture for binary skin cancer classification
- **Multiple Interfaces**: 
  - Streamlit web app for image uploads
  - Real-time camera detection
  - Location-based HTML interface
- **Data Augmentation**: Enhanced training with image transformations

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Structure

```
Skin_Data/
├── Cancer/
│   ├── Training/
│   └── Testing/
└── Non_Cancer/
    ├── Training/
    └── Testing/
```

## Usage

### 1. Train the Model
```bash
python model.py
```
This creates `skin_cancer_detection_model.h5`

### 2. Web Interface (Recommended)
```bash
streamlit run app.py
```
Upload images through the web interface for predictions.

### 3. Run locally for Detection
```bash
python predictlocally.py
```
Press 'q' to quit.

## Model Architecture

- **Input**: 224x224 RGB images
- **Layers**: 4 Conv2D blocks with BatchNormalization and MaxPooling
- **Output**: Binary classification (Cancer/Non-Cancer)
- **Activation**: Sigmoid for probability output

## Files Description

- `model.py`: Training script with CNN architecture
- `app.py`: Streamlit web application
- `predictlocally.py`: Real-time camera prediction
- `index.html`: Web interface

## Model Output

- **Probability > 0.5**: Cancer detected
- **Probability ≤ 0.5**: Non-cancer/Benign

## Disclaimer

This tool is for educational purposes only. Always consult healthcare professionals for medical diagnosis.
