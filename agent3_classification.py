import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
from agent1_preprocessing import ImagePreprocessor
from agent2_feature_extraction import FeatureExtractor


class ScratchClassifier:
    """SVM-based classifier for scratch detection."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        """
        Initialize the SVM classifier.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly').
            C: Regularization parameter.
            gamma: Kernel coefficient for 'rbf', 'poly'.
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(
            kernel=kernel, 
            C=C, 
            gamma=gamma, 
            probability=True,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize preprocessing and feature extraction
        self.preprocessor = ImagePreprocessor(target_size=(128, 64))
        self.feature_extractor = FeatureExtractor()
    
    def prepare_dataset(
        self, 
        image_paths: List[str], 
        labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset by extracting features from all images.
        
        Args:
            image_paths: List of paths to image files.
            labels: Corresponding labels (0=clean, 1=scratched).
            
        Returns:
            Feature matrix X and label array y.
        """
        features_list = []
        valid_labels = []
        
        for img_path, label in zip(image_paths, labels):
            features = self.feature_extractor.extract_from_path(
                img_path, self.preprocessor
            )
            if features is not None:
                features_list.append(features)
                valid_labels.append(label)
            else:
                print(f"Warning: Skipping {img_path} - feature extraction failed")
        
        X = np.array(features_list)
        y = np.array(valid_labels)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the SVM classifier.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Label array (n_samples,).
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"Model trained on {len(y)} samples")
        print(f"Class distribution: Clean={sum(y==0)}, Scratched={sum(y==1)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            
        Returns:
            Predicted labels.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            
        Returns:
            Class probabilities (n_samples, n_classes).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def predict_single_image(self, image_path: str) -> dict:
        """
        Predict scratch status for a single image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary with prediction and confidence.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        # Extract features
        features = self.feature_extractor.extract_from_path(
            image_path, self.preprocessor
        )
        
        if features is None:
            return {"error": "Failed to process image"}
        
        # Reshape for single sample
        X = features.reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.predict(X)[0]
        probabilities = self.predict_proba(X)[0]
        confidence = probabilities[prediction]
        
        return {
            "prediction": "scratched" if prediction == 1 else "clean",
            "confidence": round(float(confidence), 2),
            "label": int(prediction)
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': {
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.kernel = model_data['config']['kernel']
        self.C = model_data['config']['C']
        self.gamma = model_data['config']['gamma']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    import glob
    
    # Collect all images
    clean_images = sorted(glob.glob("clean/*.png"))
    scratched_images = sorted(glob.glob("scratched/*.png"))
    
    print(f"Found {len(clean_images)} clean images")
    print(f"Found {len(scratched_images)} scratched images")
    
    # Create labels
    all_images = clean_images + scratched_images
    labels = [0] * len(clean_images) + [1] * len(scratched_images)
    
    # Initialize and train classifier
    classifier = ScratchClassifier(kernel='rbf', C=1.0)
    
    # Prepare dataset
    X, y = classifier.prepare_dataset(all_images, labels)
    print(f"Dataset prepared: X shape = {X.shape}, y shape = {y.shape}")
    
    # Train model
    classifier.train(X, y)
    
    # Test prediction on a single image
    if len(scratched_images) > 0:
        result = classifier.predict_single_image(scratched_images[0])
        print(f"\nTest prediction for {scratched_images[0]}:")
        print(result)
