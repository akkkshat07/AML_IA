# Deployment Guide for Streamlit Cloud

This guide will help you deploy your Medical Image Classifier to Streamlit Cloud.

## Prerequisites

- GitHub account (you already have: akkkshat07)
- Streamlit Cloud account (sign up at https://streamlit.io/cloud)

## Steps to Deploy on Streamlit Cloud

### 1. Sign Up / Login to Streamlit Cloud

1. Go to https://streamlit.io/cloud
2. Click "Sign up" or "Sign in"
3. Use your GitHub account to authenticate

### 2. Deploy Your App

1. Once logged in, click "New app" button
2. Fill in the deployment form:
   - **Repository**: `akkkshat07/AML_IA`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
3. Click "Deploy!"

### 3. Wait for Deployment

- Streamlit Cloud will automatically:
  - Clone your repository
  - Install dependencies from `requirements.txt`
  - Start your app
- This usually takes 2-5 minutes

### 4. Access Your App

Once deployed, you'll get a URL like:
`https://akshat-aml-ia.streamlit.app`

You can share this URL with anyone!

## Important Notes

### Model File Size

- The trained model file (`models/medical_classifier_model.h5`) is included in the repo
- If GitHub complains about file size (>100MB), you may need to:
  1. Use Git LFS (Large File Storage)
  2. Or train the model on Streamlit Cloud on first run

### Training on Streamlit Cloud

If you want to train the model on Streamlit Cloud:

1. Remove the model file from git:
   ```bash
   git rm models/medical_classifier_model.h5
   git commit -m "Remove large model file"
   git push
   ```

2. Modify `streamlit_app.py` to check if model exists, and if not, run training:
   ```python
   if not os.path.exists(model_path):
       st.warning("Model not found. Training new model...")
       # Add code to run training
   ```

## Environment Variables (Optional)

If you need any environment variables:
1. Go to your app settings on Streamlit Cloud
2. Click "Advanced settings"
3. Add environment variables in the "Secrets" section

## Custom Domain (Optional)

To use a custom domain:
1. Go to app settings
2. Click "Custom subdomain"
3. Enter your desired subdomain

## Troubleshooting

### App won't start
- Check the logs in Streamlit Cloud dashboard
- Verify all dependencies are in `requirements.txt`
- Make sure Python version is compatible (3.7-3.11)

### Model loading errors
- Ensure model file is in the correct path
- Check file size limits
- Verify TensorFlow version compatibility

### Memory issues
- Streamlit Cloud free tier has memory limits
- Consider optimizing model size
- Use model quantization if needed

## Updating Your App

To update your deployed app:
```bash
# Make your changes
git add .
git commit -m "Update description"
git push
```

Streamlit Cloud will automatically detect changes and redeploy!

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [GitHub Repository](https://github.com/akkkshat07/AML_IA)

---

**Developed by Akshat**  
Applied Machine Learning Project
