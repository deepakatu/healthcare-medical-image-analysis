
"""
CNN Model for Chest X-ray Analysis
Implements a custom CNN architecture with transfer learning
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

logger = logging.getLogger(__name__)

class ChestXRayModel:
    """CNN model for chest X-ray classification"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['Normal', 'Pneumonia', 'COVID-19']
        
    def build_model(self):
        """Build CNN model with transfer learning"""
        try:
            # Base model (ResNet50)
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Freeze base model layers
            base_model.trainable = False
            
            # Add custom classification head
            model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'precision', 'recall']
            )
            
            self.model = model
            logger.info("Model built successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def train(self, train_data, validation_data, epochs=50, batch_size=32):
        """Train the model"""
        try:
            if self.model is None:
                self.build_model()
            
            # Data augmentation
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
            
            val_datagen = ImageDataGenerator(rescale=1./255)
            
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'models/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max'
                )
            ]
            
            # Train model
            history = self.model.fit(
                train_datagen.flow_from_directory(
                    train_data,
                    target_size=self.input_shape[:2],
                    batch_size=batch_size,
                    class_mode='categorical'
                ),
                epochs=epochs,
                validation_data=val_datagen.flow_from_directory(
                    validation_data,
                    target_size=self.input_shape[:2],
                    batch_size=batch_size,
                    class_mode='categorical'
                ),
                callbacks=callbacks
            )
            
            logger.info("Model training completed")
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, image):
        """Make prediction on preprocessed image"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Ensure image has correct shape
            if len(image.shape) == 3:
                image = np.expand_dims(image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(image, verbose=0)
            
            # Get class with highest probability
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            return predicted_class, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def predict_with_uncertainty(self, image, n_samples=10):
        """Make prediction with uncertainty estimation using Monte Carlo Dropout"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_model() first.")
            
            # Enable dropout during inference
            predictions = []
            for _ in range(n_samples):
                pred = self.model(image, training=True)
                predictions.append(pred.numpy())
            
            predictions = np.array(predictions)
            
            # Calculate mean and standard deviation
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Get final prediction
            predicted_class_idx = np.argmax(mean_pred[0])
            confidence = float(mean_pred[0][predicted_class_idx])
            uncertainty = float(std_pred[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            return predicted_class, confidence, uncertainty
            
        except Exception as e:
            logger.error(f"Error making uncertainty prediction: {str(e)}")
            raise
    
    def load_model(self, model_path='models/chest_xray_model.h5'):
        """Load pre-trained model"""
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                # Build and use a dummy trained model for demo
                logger.warning(f"Model file {model_path} not found. Building new model.")
                self.build_model()
                self._create_dummy_weights()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            # Fallback to building new model
            self.build_model()
            self._create_dummy_weights()
    
    def save_model(self, model_path='models/chest_xray_model.h5'):
        """Save trained model"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
            
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def _create_dummy_weights(self):
        """Create dummy weights for demonstration purposes"""
        try:
            # Create dummy data to initialize model weights
            dummy_input = np.random.random((1, *self.input_shape))
            dummy_output = np.random.random((1, self.num_classes))
            
            # Compile and fit with dummy data
            self.model.fit(dummy_input, dummy_output, epochs=1, verbose=0)
            logger.info("Dummy weights created for demonstration")
            
        except Exception as e:
            logger.error(f"Error creating dummy weights: {str(e)}")
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        
        return buffer.getvalue()
    
    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            test_datagen = ImageDataGenerator(rescale=1./255)
            test_generator = test_datagen.flow_from_directory(
                test_data,
                target_size=self.input_shape[:2],
                batch_size=32,
                class_mode='categorical',
                shuffle=False
            )
            
            # Evaluate
            results = self.model.evaluate(test_generator, verbose=0)
            
            # Create results dictionary
            metrics = {}
            for i, metric in enumerate(self.model.metrics_names):
                metrics[metric] = results[i]
            
            logger.info(f"Model evaluation completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
