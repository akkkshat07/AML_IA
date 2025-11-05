# Medical Image Classifier - AML Project

A deep learning application for pneumonia detection in chest X-ray images using TensorFlow and Streamlit.

**Developed by: Akshat**

## Project Overview

This computer vision project showcases a complete CNN implementation for medical image classification. It includes a training pipeline, model evaluation, and an interactive web interface for image upload and prediction. This project was developed as part of the Applied Machine Learning coursework.

## Features

- **Deep Learning Model**: CNN architecture for binary image classification
- **Training Pipeline**: Complete model training with evaluation metrics
- **Web Interface**: Streamlit app for image upload and prediction
- **Model Persistence**: Save and load trained models
- **Error Handling**: Robust error handling throughout the pipeline

## Tech Stack

- **Deep Learning**: TensorFlow, Keras
- **Computer Vision**: OpenCV, Pillow
- **Web Framework**: Streamlit
- **Data Science**: NumPy, Pandas, Matplotlib
- **Model Evaluation**: scikit-learn

## Project Structure

```
medical-image-classifier/
├── data/
│   ├── train/              # Training images folder
│   ├── test/               # Test images folder
│   └── val/                # Validation images folder
├── models/                 # Model storage
├── src/
│   └── model_training.py   # CNN training script
├── streamlit_app.py        # Web interface
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/akkkshat07/AML_IA.git
cd AML_IA
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python src/model_training.py
```

### Running the Web Application
```bash
streamlit run streamlit_app.py
```
Visit `http://localhost:8501` to access the interface.

## Model Implementation

- **Architecture**: CNN with convolutional blocks and global average pooling
- **Training**: Uses sample data for demonstration purposes
- **Evaluation**: Provides accuracy, precision, recall, and F1-score metrics
- **Web Interface**: Allows image upload with real-time predictions

## Technical Details

### Model Architecture
- Convolutional layers with ReLU activation
- Batch normalization for stable training
- Dropout layers for regularization
- Binary classification output with sigmoid activation

### Training Features
- Early stopping to prevent overfitting
- Model checkpointing for best weights
- Comprehensive evaluation metrics
- Training history visualization

## Development Notes

This project currently uses synthetic sample data to demonstrate the complete ML pipeline. The architecture and training process are designed to work with real medical imaging datasets when integrated.

## Medical Disclaimer

This tool is for educational and demonstration purposes only. It should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## Future Enhancements

- Integration with real medical imaging datasets
- Docker containerization for deployment
- Additional preprocessing techniques
- Model performance optimization
- Extended evaluation metrics
- Hyperparameter tuning
- Data augmentation improvements

## Academic Context

This project was developed as part of the Applied Machine Learning (AML) course to demonstrate practical implementation of deep learning techniques in medical image analysis.

## Contact

**Akshat**
- GitHub: [@akkkshat07](https://github.com/akkkshat07)
- Project Repository: [AML_IA](https://github.com/akkkshat07/AML_IA.git)
