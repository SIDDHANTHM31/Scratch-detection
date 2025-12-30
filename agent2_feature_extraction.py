import numpy as np
from skimage.feature import hog
from typing import Tuple, Optional
from agent1_preprocessing import ImagePreprocessor


class FeatureExtractor:
    """Extracts HOG features from preprocessed document images."""
    
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        block_norm: str = 'L2-Hys'
    ):
        """
        Initialize the HOG feature extractor.
        
        Args:
            orientations: Number of orientation bins for HOG.
            pixels_per_cell: Size (in pixels) of a cell.
            cells_per_block: Number of cells in each block.
            block_norm: Block normalization method.
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        
    def extract_hog_features(
        self, 
        image: np.ndarray, 
        visualize: bool = False
    ) -> np.ndarray:
        """
        Extract HOG features from a preprocessed image.
        
        Args:
            image: Preprocessed grayscale image.
            visualize: Whether to return HOG visualization image.
            
        Returns:
            HOG feature vector (and visualization if requested).
        """
        if visualize:
            features, hog_image = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=True,
                feature_vector=True
            )
            return features, hog_image
        else:
            features = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=False,
                feature_vector=True
            )
            return features
    
    def extract_from_path(
        self, 
        image_path: str, 
        preprocessor: ImagePreprocessor
    ) -> Optional[np.ndarray]:
        """
        Extract HOG features directly from an image path.
        
        Args:
            image_path: Path to the image file.
            preprocessor: ImagePreprocessor instance.
            
        Returns:
            HOG feature vector, or None if extraction fails.
        """
        # Preprocess the image
        processed_image = preprocessor.preprocess_for_hog(image_path)
        if processed_image is None:
            return None
        
        # Extract HOG features
        features = self.extract_hog_features(processed_image)
        return features
    
    def get_feature_dimension(self, image_shape: Tuple[int, int]) -> int:
        """
        Calculate the expected HOG feature vector dimension.
        
        Args:
            image_shape: Shape of the input image (height, width).
            
        Returns:
            Expected feature vector length.
        """
        height, width = image_shape
        
        # Number of cells
        n_cells_x = width // self.pixels_per_cell[0]
        n_cells_y = height // self.pixels_per_cell[1]
        
        # Number of blocks
        n_blocks_x = n_cells_x - self.cells_per_block[0] + 1
        n_blocks_y = n_cells_y - self.cells_per_block[1] + 1
        
        # Feature vector length
        feature_dim = (
            n_blocks_x * n_blocks_y * 
            self.cells_per_block[0] * self.cells_per_block[1] * 
            self.orientations
        )
        
        return feature_dim


if __name__ == "__main__":
    # Test the feature extractor
    preprocessor = ImagePreprocessor(target_size=(128, 64))
    extractor = FeatureExtractor()
    
    # Test with sample images
    test_images = [
        "clean/clean_0.png",
        "scratched/scratched_0.png"
    ]
    
    for img_path in test_images:
        features = extractor.extract_from_path(img_path, preprocessor)
        if features is not None:
            print(f"Extracted from {img_path}: feature_dim={len(features)}")
        else:
            print(f"Failed to extract from {img_path}")
    
    # Print expected feature dimension
    expected_dim = extractor.get_feature_dimension((64, 128))
    print(f"\nExpected feature dimension for (64, 128) image: {expected_dim}")
