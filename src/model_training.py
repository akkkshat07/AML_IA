"""
Medical Image Classifier - CNN Model Training (Production Ready)
Applied Machine Learning Project
Author: Akshat
"""

import os
import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix

# Handle TensorFlow imports with version compatibility
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("Error: TensorFlow not installed. Run: pip install tensorflow")
    exit(1)

class MedicalImageClassifier:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        
    def create_model(self):
        """Create CNN architecture for medical image classification"""
        try:
            model = keras.Sequential([
                # Input layer
                layers.Input(shape=(self.img_height, self.img_width, 3)),
                
                # First conv block
                layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Second conv block
                layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Third conv block
                layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Classifier
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(1, activation='sigmoid')  # Binary classification
            ])
            
            # Compile model
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            print("Model created successfully")
            return model
            
        except Exception as e:
            print(f"Error creating model: {e}")
            return None
    
    def load_real_data(self, data_dir='data'):
        """Load real X-ray images from directory structure"""
        print("Loading real chest X-ray dataset...")
        
        import os
        from PIL import Image
        import cv2
        
        def load_images_from_folder(folder_path, label):
            """Load all images from a folder with given label"""
            images = []
            labels = []
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist")
                return images, labels
            
            file_count = 0
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, filename)
                    try:
                        # Load and resize image
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (self.img_height, self.img_width))
                            images.append(img)
                            labels.append(label)
                            file_count += 1
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
            
            print(f"  Loaded {file_count} images from {folder_path}")
            return images, labels
        
        # Get script directory and construct data paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        data_path = os.path.join(project_dir, data_dir)
        
        # Load training data
        print("\nLoading training data...")
        train_normal_imgs, train_normal_labels = load_images_from_folder(
            os.path.join(data_path, 'train', 'NORMAL'), 0
        )
        train_pneumonia_imgs, train_pneumonia_labels = load_images_from_folder(
            os.path.join(data_path, 'train', 'PNEUMONIA'), 1
        )
        
        # Load validation data
        print("\nLoading validation data...")
        val_normal_imgs, val_normal_labels = load_images_from_folder(
            os.path.join(data_path, 'val', 'NORMAL'), 0
        )
        val_pneumonia_imgs, val_pneumonia_labels = load_images_from_folder(
            os.path.join(data_path, 'val', 'PNEUMONIA'), 1
        )
        
        # Load test data
        print("\nLoading test data...")
        test_normal_imgs, test_normal_labels = load_images_from_folder(
            os.path.join(data_path, 'test', 'NORMAL'), 0
        )
        test_pneumonia_imgs, test_pneumonia_labels = load_images_from_folder(
            os.path.join(data_path, 'test', 'PNEUMONIA'), 1
        )
        
        # Combine and convert to numpy arrays
        X_train = np.array(train_normal_imgs + train_pneumonia_imgs, dtype=np.float32) / 255.0
        y_train = np.array(train_normal_labels + train_pneumonia_labels)
        
        X_val = np.array(val_normal_imgs + val_pneumonia_imgs, dtype=np.float32) / 255.0
        y_val = np.array(val_normal_labels + val_pneumonia_labels)
        
        X_test = np.array(test_normal_imgs + test_pneumonia_imgs, dtype=np.float32) / 255.0
        y_test = np.array(test_normal_labels + test_pneumonia_labels)
        
        # Shuffle training data
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        print(f"\n{'='*50}")
        print("Dataset loaded successfully!")
        print(f"{'='*50}")
        print(f"Training: {X_train.shape}, Labels: {y_train.shape}")
        print(f"  - Normal: {np.sum(y_train == 0)}, Pneumonia: {np.sum(y_train == 1)}")
        print(f"Validation: {X_val.shape}, Labels: {y_val.shape}")
        print(f"  - Normal: {np.sum(y_val == 0)}, Pneumonia: {np.sum(y_val == 1)}")
        print(f"Test: {X_test.shape}, Labels: {y_test.shape}")
        print(f"  - Normal: {np.sum(y_test == 0)}, Pneumonia: {np.sum(y_test == 1)}")
        print(f"{'='*50}\n")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def train_model(self, train_data, val_data, epochs=20):
        """Train the CNN model"""
        if self.model is None:
            print("Error: Model not created. Call create_model() first.")
            return
            
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print("Starting model training...")
        
        try:
            # Calculate appropriate batch size
            batch_size = 32
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=4,
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    verbose=1,
                    min_lr=0.00001
                )
            ]
            
            # Train model
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            print("Model training completed!")
            
        except Exception as e:
            print(f"Error during training: {e}")
            self.history = None
        
    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        if self.model is None:
            print("Error: No trained model available")
            return None
            
        X_test, y_test = test_data
        
        print("Evaluating model performance...")
        
        try:
            # Get predictions
            predictions = self.model.predict(X_test, verbose=0)
            y_pred = (predictions > 0.5).astype(int).flatten()
            
            # Calculate basic metrics
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Calculate additional metrics manually
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Print results
            print("\n" + "="*50)
            print("MODEL PERFORMANCE RESULTS")
            print("="*50)
            print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Test Loss: {test_loss:.4f}")
            
            # Classification report
            print("\nDETAILED CLASSIFICATION REPORT:")
            print("-" * 50)
            try:
                print(classification_report(y_test, y_pred, target_names=['Normal', 'Pneumonia']))
            except:
                print("Classification report not available")
            
            return {
                'accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'loss': test_loss
            }
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-GUI backend
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # Accuracy
            axes[0].plot(self.history.history['accuracy'], label='Training')
            axes[0].plot(self.history.history['val_accuracy'], label='Validation')
            axes[0].set_title('Model Accuracy')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Accuracy')
            axes[0].legend()
            
            # Loss
            axes[1].plot(self.history.history['loss'], label='Training')
            axes[1].plot(self.history.history['val_loss'], label='Validation')
            axes[1].set_title('Model Loss')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].legend()
            
            plt.tight_layout()
            
            # Get correct path
            script_dir = os.path.dirname(__file__)
            models_dir = os.path.join(os.path.dirname(script_dir), 'models')
            os.makedirs(models_dir, exist_ok=True)
            plot_path = os.path.join(models_dir, 'training_history.png')
            
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Training plots saved to: {plot_path}")
            
        except Exception as e:
            print(f"Could not create plots: {e}")
    
    def save_model(self):
        """Save the trained model"""
        if self.model is None:
            print("No model to save")
            return
            
        try:
            # Get correct paths
            script_dir = os.path.dirname(__file__)
            models_dir = os.path.join(os.path.dirname(script_dir), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, 'medical_classifier_model.h5')
            summary_path = os.path.join(models_dir, 'model_summary.txt')
            
            # Save model
            self.model.save(model_path)
            
            # Save model summary
            with open(summary_path, 'w') as f:
                self.model.summary(print_fn=lambda x: f.write(x + '\n'))
            
            print(f"Model saved to: {model_path}")
            print(f"Model summary saved to: {summary_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")

def main():
    """Main training pipeline"""
    print("MEDICAL IMAGE CLASSIFIER TRAINING")
    print("=" * 50)
    
    try:
        # Initialize classifier
        classifier = MedicalImageClassifier()
        
        # Create model
        print("Creating CNN model...")
        model = classifier.create_model()
        
        if model is None:
            print("Failed to create model. Exiting.")
            return
        
        # Print model summary
        print("\nModel Architecture:")
        print("-" * 30)
        model.summary()
        
        # Load real dataset
        print("\nPreparing dataset...")
        train_data, val_data, test_data = classifier.load_real_data()
        
        # Train model
        print("\nTraining model...")
        classifier.train_model(train_data, val_data, epochs=20)
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = classifier.evaluate_model(test_data)
        
        # Plot and save
        classifier.plot_training_history()
        classifier.save_model()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED!")
        print("="*50)
        print("Model trained on real chest X-ray dataset.")
        print("Model saved and ready for deployment!")
        
        return metrics
        
    except Exception as e:
        print(f"Training failed: {e}")
        return None

if __name__ == "__main__":
    main()
