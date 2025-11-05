"""
Medical Image Classifier - Streamlit Web Application
Applied Machine Learning Project
Author: Akshat
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Medical Image Classifier",
    page_icon="🏥",
    layout="wide"
)

def load_model():
    """Load the trained model"""
    try:
        import tensorflow as tf
        model_path = "models/medical_classifier_model.h5"
        
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess uploaded image for prediction"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize pixel values
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def predict_pneumonia(model, image):
    """Make prediction on uploaded image"""
    try:
        preprocessed_image = preprocess_image(image)
        
        if preprocessed_image is None:
            return None
        
        # Make prediction
        prediction = model.predict(preprocessed_image, verbose=0)
        confidence = float(prediction[0][0])
        
        # Interpret prediction
        if confidence > 0.5:
            predicted_class = "Pneumonia"
            confidence_percent = confidence * 100
        else:
            predicted_class = "Normal"
            confidence_percent = (1 - confidence) * 100
        
        return {
            'class': predicted_class,
            'confidence': confidence_percent,
            'raw_prediction': confidence
        }
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🏥 Medical Image Classifier")
    st.subheader("Pneumonia Detection in Chest X-rays")
    st.caption("Applied Machine Learning Project by Akshat")
    
    # Sidebar
    st.sidebar.header("About This Tool")
    st.sidebar.info(
        "This application uses a convolutional neural network (CNN) "
        "to analyze chest X-ray images and detect signs of pneumonia. "
        "Upload an X-ray image to get a prediction."
    )
    
    st.sidebar.warning(
        "⚠️ **Medical Disclaimer**: This tool is for educational purposes only. "
        "It should not be used as a substitute for professional medical diagnosis. "
        "Always consult qualified healthcare professionals for medical decisions."
    )
    
    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Model not found!")
        st.info("Please run the training script first: `python src/model_training.py`")
        st.stop()
    else:
        st.success("✅ AI model loaded successfully!")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            # Image info
            st.write(f"**Image size:** {image.size}")
            st.write(f"**File size:** {uploaded_file.size} bytes")
    
    with col2:
        st.header("🔬 Analysis Results")
        
        if uploaded_file is not None:
            # Make prediction
            with st.spinner("Analyzing image..."):
                result = predict_pneumonia(model, image)
            
            if result is not None:
                # Display results
                if result['class'] == "Pneumonia":
                    st.error(f"🚨 **Prediction: {result['class']}**")
                    st.write(f"**Confidence:** {result['confidence']:.1f}%")
                else:
                    st.success(f"✅ **Prediction: {result['class']}**")
                    st.write(f"**Confidence:** {result['confidence']:.1f}%")
                
                # Confidence bar
                confidence_bar = result['confidence'] / 100
                st.progress(confidence_bar)
                
                # Additional information
                st.subheader("📊 Technical Details")
                st.write(f"**Raw prediction score:** {result['raw_prediction']:.4f}")
                st.write(f"**Threshold:** 0.5")
                st.write(f"**Model architecture:** CNN with 4 convolutional blocks")
                
                # Interpretation guide
                st.subheader("📋 How to Interpret")
                st.write("""
                - **Normal**: No signs of pneumonia detected
                - **Pneumonia**: Potential signs of pneumonia detected
                - **Confidence**: How certain the model is about its prediction
                - **Higher confidence** indicates more certainty in the prediction
                """)
        else:
            st.info("👆 Please upload a chest X-ray image to get started")
    
    # Additional sections
    st.header("🧠 About the AI Model")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.metric("Model Type", "CNN")
        st.write("Convolutional Neural Network optimized for medical imaging")
    
    with col4:
        st.metric("Training Images", "~6,000")
        st.write("Trained on diverse chest X-ray dataset")
    
    with col5:
        st.metric("Accuracy", "~90%")
        st.write("Performance on test dataset")
    
    # Model architecture
    if st.expander("🏗️ Model Architecture Details"):
        st.write("""
        **Layer Structure:**
        1. Input Layer (224x224x3)
        2. Conv2D + BatchNorm + MaxPool + Dropout (32 filters)
        3. Conv2D + BatchNorm + MaxPool + Dropout (64 filters)  
        4. Conv2D + BatchNorm + MaxPool + Dropout (128 filters)
        5. GlobalAveragePooling2D
        6. Dense + BatchNorm + Dropout (256 units)
        7. Dense Output (1 unit, sigmoid activation)
        
        **Training Details:**
        - Optimizer: Adam (learning_rate=0.001)
        - Loss: Binary Crossentropy
        - Data Augmentation: Applied during training
        - Early Stopping: Prevents overfitting
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Developed by Akshat** | "
        "Applied Machine Learning Project | "
        "[GitHub Repository](https://github.com/akkkshat07/AML_IA)"
    )

if __name__ == "__main__":
    main()
