# Model Training Issue & Solution

## Problem
The current model predicts everything as "Normal" because it was trained on random synthetic data that doesn't represent real X-ray patterns.

## Why This Happens
1. The original training data was completely random pixels
2. The model couldn't learn meaningful patterns
3. It defaulted to predicting the majority class (Normal)

## Solutions

### Option 1: Use Real Dataset (Recommended for Production)
Download a real chest X-ray dataset like:
- **Kaggle Chest X-Ray Pneumonia Dataset**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Contains ~5,800 actual chest X-ray images
- Organized into train/test/val folders

Steps:
```bash
# Download dataset from Kaggle
# Place images in:
# - data/train/NORMAL/
# - data/train/PNEUMONIA/
# - data/test/NORMAL/
# - data/test/PNEUMONIA/  
# - data/val/NORMAL/
# - data/val/PNEUMONIA/

# Then update src/model_training.py to load from these folders instead of generating synthetic data
```

### Option 2: Improved Synthetic Data (Current Implementation)
The updated `model_training.py` now creates better synthetic data with:
- Balanced classes (50% Normal, 50% Pneumonia)
- Realistic patterns (white patches for pneumonia, clearer for normal)
- Proper data distribution

To retrain with improved data:
```bash
python3 src/model_training.py
```

Wait for training to complete (~5-10 minutes on M1/M2 Mac).

### Option 3: Quick Demo Fix
For immediate demonstration purposes, you can:

1. **Lower the threshold** in predictions
2. **Add randomness** to make it less biased
3. **Show both scores** instead of binary prediction

## Current Status

The improved training script (`src/model_training.py`) has been updated with:
- ✅ Better synthetic data generation
- ✅ Balanced class distribution
- ✅ Realistic X-ray-like patterns
- ✅ Reduced training time
- ✅ Better callbacks and learning rate scheduling

## Next Steps

1. **Complete the training**: Run `python3 src/model_training.py` and let it finish
2. **The model will be saved** to `models/medical_classifier_model.h5`
3. **Restart Streamlit** to load the new model
4. **Test with images** - it should now predict both classes

## For Best Results

Consider using a real dataset for production use. The synthetic data is only for demonstration and will never be as accurate as real medical images.

---

**Note**: Training was interrupted. Please run the training command again and let it complete fully for the improved model to work properly.
