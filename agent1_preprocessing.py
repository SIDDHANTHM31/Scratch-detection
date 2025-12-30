import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """Preprocesses document images for scratch detection."""
    
    def __init__(self, target_size: Tuple[int, int] = (128, 64)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size: Target dimensions (width, height) for resized images.
        """
        self.target_size = target_size
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from disk.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Loaded image as numpy array, or None if loading fails.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return None
        return image
    
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input BGR image.
            
        Returns:
            Grayscale image.
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply Gaussian blur to remove noise.
        
        Args:
            image: Input grayscale image.
            kernel_size: Size of the Gaussian kernel (must be odd).
            
        Returns:
            Blurred image.
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding to create binary image.
        
        Args:
            image: Input grayscale image.
            
        Returns:
            Binary thresholded image.
        """
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image.
            
        Returns:
            Resized image.
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1] range.
        
        Args:
            image: Input image.
            
        Returns:
            Normalized image as float32.
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess(self, image_path: str) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Args:
            image_path: Path to the input image.
            
        Returns:
            Preprocessed image ready for feature extraction, or None if failed.
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Apply Gaussian blur to remove noise
        blurred = self.apply_gaussian_blur(gray)
        
        # Apply adaptive thresholding
        binary = self.apply_adaptive_threshold(blurred)
        
        # Resize to fixed dimensions
        resized = self.resize_image(binary)
        
        # Normalize pixel values
        normalized = self.normalize_image(resized)
        
        return normalized
    
    def preprocess_for_hog(self, image_path: str) -> Optional[np.ndarray]:
        """
        Preprocess image specifically for HOG feature extraction.
        HOG works better with grayscale images without binarization.
        
        Args:
            image_path: Path to the input image.
            
        Returns:
            Preprocessed grayscale image for HOG extraction.
        """
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # Convert to grayscale
        gray = self.convert_to_grayscale(image)
        
        # Apply light Gaussian blur to reduce noise
        blurred = self.apply_gaussian_blur(gray, kernel_size=3)
        
        # Resize to fixed dimensions
        resized = self.resize_image(blurred)
        
        return resized


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = ImagePreprocessor(target_size=(128, 64))
    
    # Test with sample images
    test_images = [
        "clean/clean_0.png",
        "scratched/scratched_0.png"
    ]
    
    for img_path in test_images:
        result = preprocessor.preprocess_for_hog(img_path)
        if result is not None:
            print(f"Processed {img_path}: shape={result.shape}, dtype={result.dtype}")
        else:
            print(f"Failed to process {img_path}")
