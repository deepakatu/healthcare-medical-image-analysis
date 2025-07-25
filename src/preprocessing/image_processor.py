
"""
Image preprocessing utilities for medical images
Handles DICOM, PNG, JPG formats with medical-specific preprocessing
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pydicom
import os
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Medical image preprocessing class"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def preprocess_image(self, image_path):
        """Main preprocessing pipeline"""
        try:
            # Load image based on format
            if image_path.lower().endswith(('.dcm', '.dicom')):
                image = self._load_dicom(image_path)
            else:
                image = self._load_standard_image(image_path)
            
            # Apply preprocessing steps
            image = self._normalize_intensity(image)
            image = self._enhance_contrast(image)
            image = self._resize_image(image)
            image = self._apply_medical_filters(image)
            image = self._normalize_for_model(image)
            
            logger.info(f"Image preprocessed successfully: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def _load_dicom(self, dicom_path):
        """Load DICOM image"""
        try:
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array
            
            # Handle different DICOM formats
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply DICOM-specific transformations
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                center = dicom_data.WindowCenter
                width = dicom_data.WindowWidth
                image = self._apply_windowing(image, center, width)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading DICOM: {str(e)}")
            raise
    
    def _load_standard_image(self, image_path):
        """Load standard image formats (PNG, JPG)"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading standard image: {str(e)}")
            raise
    
    def _apply_windowing(self, image, center, width):
        """Apply DICOM windowing"""
        try:
            min_val = center - width // 2
            max_val = center + width // 2
            
            image = np.clip(image, min_val, max_val)
            image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            logger.error(f"Error applying windowing: {str(e)}")
            return image
    
    def _normalize_intensity(self, image):
        """Normalize image intensity"""
        try:
            # Convert to float
            image = image.astype(np.float32)
            
            # Normalize to 0-255 range
            image = (image - image.min()) / (image.max() - image.min()) * 255
            
            return image.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error normalizing intensity: {str(e)}")
            return image
    
    def _enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing contrast: {str(e)}")
            return image
    
    def _resize_image(self, image):
        """Resize image to target size"""
        try:
            resized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            return resized
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return image
    
    def _apply_medical_filters(self, image):
        """Apply medical-specific filters"""
        try:
            # Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Unsharp masking for edge enhancement
            unsharp_mask = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
            
            # Ensure values are in valid range
            unsharp_mask = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
            
            return unsharp_mask
            
        except Exception as e:
            logger.error(f"Error applying medical filters: {str(e)}")
            return image
    
    def _normalize_for_model(self, image):
        """Normalize image for model input"""
        try:
            # Convert to RGB (3 channels) for model compatibility
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Normalize to 0-1 range
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error normalizing for model: {str(e)}")
            return image
    
    def extract_roi(self, image, method='lung_segmentation'):
        """Extract region of interest (ROI)"""
        try:
            if method == 'lung_segmentation':
                return self._segment_lungs(image)
            elif method == 'chest_detection':
                return self._detect_chest_area(image)
            else:
                return image
                
        except Exception as e:
            logger.error(f"Error extracting ROI: {str(e)}")
            return image
    
    def _segment_lungs(self, image):
        """Segment lung regions using thresholding and morphology"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour (assumed to be lungs)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create mask
                mask = np.zeros_like(gray)
                cv2.fillPoly(mask, [largest_contour], 255)
                
                # Apply mask
                segmented = cv2.bitwise_and(image, image, mask=mask)
                return segmented
            
            return image
            
        except Exception as e:
            logger.error(f"Error segmenting lungs: {str(e)}")
            return image
    
    def _detect_chest_area(self, image):
        """Detect and crop chest area"""
        try:
            # Simple chest detection using image statistics
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Find non-zero regions
            coords = cv2.findNonZero(gray)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                # Crop image
                cropped = image[y:y+h, x:x+w]
                return cropped
            
            return image
            
        except Exception as e:
            logger.error(f"Error detecting chest area: {str(e)}")
            return image
    
    def augment_image(self, image, augmentation_type='random'):
        """Apply data augmentation"""
        try:
            if augmentation_type == 'rotation':
                angle = np.random.uniform(-15, 15)
                center = (image.shape[1] // 2, image.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                augmented = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
                
            elif augmentation_type == 'brightness':
                factor = np.random.uniform(0.8, 1.2)
                augmented = cv2.convertScaleAbs(image, alpha=factor, beta=0)
                
            elif augmentation_type == 'noise':
                noise = np.random.normal(0, 0.1, image.shape)
                augmented = np.clip(image + noise, 0, 1)
                
            else:  # random
                augmentations = ['rotation', 'brightness', 'noise']
                chosen = np.random.choice(augmentations)
                return self.augment_image(image, chosen)
            
            return augmented
            
        except Exception as e:
            logger.error(f"Error augmenting image: {str(e)}")
            return image
    
    def get_image_statistics(self, image):
        """Get image statistics for analysis"""
        try:
            stats = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'min_value': float(np.min(image)),
                'max_value': float(np.max(image)),
                'mean_value': float(np.mean(image)),
                'std_value': float(np.std(image)),
                'unique_values': len(np.unique(image))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting image statistics: {str(e)}")
            return {}
