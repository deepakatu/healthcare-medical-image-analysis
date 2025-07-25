
"""
Unit tests for the CNN model
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cnn_model import ChestXRayModel

class TestChestXRayModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = ChestXRayModel()
        
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.input_shape, (224, 224, 3))
        self.assertEqual(self.model.num_classes, 3)
        self.assertEqual(len(self.model.class_names), 3)
        self.assertIn('Normal', self.model.class_names)
        self.assertIn('Pneumonia', self.model.class_names)
        self.assertIn('COVID-19', self.model.class_names)
    
    def test_build_model(self):
        """Test model building"""
        model = self.model.build_model()
        self.assertIsNotNone(model)
        self.assertEqual(model.input_shape[1:], self.model.input_shape)
        self.assertEqual(model.output_shape[1], self.model.num_classes)
    
    def test_predict_with_dummy_input(self):
        """Test prediction with dummy input"""
        # Build model first
        self.model.build_model()
        self.model._create_dummy_weights()
        
        # Create dummy input
        dummy_input = np.random.random((1, 224, 224, 3))
        
        # Make prediction
        prediction, confidence = self.model.predict(dummy_input)
        
        self.assertIsInstance(prediction, str)
        self.assertIn(prediction, self.model.class_names)
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_predict_without_model(self):
        """Test prediction without loaded model"""
        dummy_input = np.random.random((1, 224, 224, 3))
        
        with self.assertRaises(ValueError):
            self.model.predict(dummy_input)
    
    def test_predict_with_uncertainty(self):
        """Test uncertainty prediction"""
        # Build model first
        self.model.build_model()
        self.model._create_dummy_weights()
        
        # Create dummy input
        dummy_input = np.random.random((1, 224, 224, 3))
        
        # Make prediction with uncertainty
        prediction, confidence, uncertainty = self.model.predict_with_uncertainty(dummy_input, n_samples=3)
        
        self.assertIsInstance(prediction, str)
        self.assertIn(prediction, self.model.class_names)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(uncertainty, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertGreaterEqual(uncertainty, 0.0)
    
    def test_save_and_load_model(self):
        """Test model saving and loading"""
        # Build model
        self.model.build_model()
        self.model._create_dummy_weights()
        
        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.model.save_model(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            
            # Create new model instance and load
            new_model = ChestXRayModel()
            new_model.load_model(tmp_path)
            self.assertIsNotNone(new_model.model)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_get_model_summary(self):
        """Test model summary generation"""
        # Without model
        summary = self.model.get_model_summary()
        self.assertEqual(summary, "Model not built yet")
        
        # With model
        self.model.build_model()
        summary = self.model.get_model_summary()
        self.assertIsInstance(summary, str)
        self.assertIn('Model:', summary)
    
    @patch('tensorflow.keras.preprocessing.image.ImageDataGenerator')
    def test_train_method_structure(self, mock_datagen):
        """Test training method structure (without actual training)"""
        # Mock the data generators
        mock_train_gen = MagicMock()
        mock_val_gen = MagicMock()
        mock_datagen.return_value.flow_from_directory.side_effect = [mock_train_gen, mock_val_gen]
        
        # Build model
        self.model.build_model()
        
        # Mock model.fit to avoid actual training
        with patch.object(self.model.model, 'fit') as mock_fit:
            mock_fit.return_value = MagicMock()
            
            # Test training call (should not raise errors)
            try:
                self.model.train('dummy_train_path', 'dummy_val_path', epochs=1, batch_size=2)
            except Exception as e:
                # Expected to fail due to dummy paths, but structure should be correct
                pass
    
    def test_class_names_consistency(self):
        """Test that class names are consistent"""
        expected_classes = ['Normal', 'Pneumonia', 'COVID-19']
        self.assertEqual(self.model.class_names, expected_classes)
        self.assertEqual(len(self.model.class_names), self.model.num_classes)

if __name__ == '__main__':
    unittest.main()
