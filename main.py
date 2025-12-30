"""
Scratch Detection - Main Pipeline
===================================
A complete machine learning pipeline for detecting scratched/crossed-out 
words in scanned document images.

This script orchestrates all 4 agents:
- Agent 1: Image Preprocessing
- Agent 2: Feature Extraction (HOG)
- Agent 3: Classification (SVM)
- Agent 4: Evaluation

Usage:
    python main.py                    # Train and evaluate on dataset
    python main.py --predict <image>  # Predict on a single image
"""

import os
import glob
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# Import all agents
from agent1_preprocessing import ImagePreprocessor
from agent2_feature_extraction import FeatureExtractor
from agent3_classification import ScratchClassifier
from agent4_evaluation import ModelEvaluator


class ScratchDetectionPipeline:
    """Complete pipeline for scratch detection."""
    
    def __init__(self, model_path: str = "scratch_model.pkl"):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to save/load the trained model.
        """
        self.model_path = model_path
        self.classifier = ScratchClassifier(kernel='rbf', C=1.0)
        self.evaluator = ModelEvaluator()
    
    def load_dataset(
        self, 
        clean_dir: str = "clean", 
        scratched_dir: str = "scratched"
    ) -> tuple:
        """
        Load dataset from directory structure.
        
        Args:
            clean_dir: Directory containing clean text images.
            scratched_dir: Directory containing scratched text images.
            
        Returns:
            Tuple of (image_paths, labels).
        """
        # Find all images
        clean_images = sorted(glob.glob(os.path.join(clean_dir, "*.png")))
        scratched_images = sorted(glob.glob(os.path.join(scratched_dir, "*.png")))
        
        # Also check for jpg images
        clean_images += sorted(glob.glob(os.path.join(clean_dir, "*.jpg")))
        scratched_images += sorted(glob.glob(os.path.join(scratched_dir, "*.jpg")))
        
        print(f"📁 Dataset loaded:")
        print(f"   - Clean images: {len(clean_images)}")
        print(f"   - Scratched images: {len(scratched_images)}")
        print(f"   - Total: {len(clean_images) + len(scratched_images)}")
        
        # Combine and create labels
        all_images = clean_images + scratched_images
        labels = [0] * len(clean_images) + [1] * len(scratched_images)
        
        return all_images, labels
    
    def train_and_evaluate(
        self, 
        image_paths: list, 
        labels: list,
        test_size: float = 0.3
    ) -> dict:
        """
        Train the model and evaluate performance.
        
        Args:
            image_paths: List of image file paths.
            labels: Corresponding labels.
            test_size: Fraction of data to use for testing.
            
        Returns:
            Evaluation results dictionary.
        """
        print("\n" + "=" * 60)
        print("SCRATCH DETECTION PIPELINE - TRAINING & EVALUATION")
        print("=" * 60)
        
        # Step 1: Prepare dataset (preprocessing + feature extraction)
        print("\n🔄 Step 1: Preparing dataset (preprocessing + feature extraction)...")
        X, y = self.classifier.prepare_dataset(image_paths, labels)
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        
        # Step 2: Split into train/test sets
        print(f"\n🔄 Step 2: Splitting dataset (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y
        )
        print(f"   Training samples: {len(y_train)}")
        print(f"   Testing samples: {len(y_test)}")
        
        # Step 3: Train the classifier
        print("\n🔄 Step 3: Training SVM classifier...")
        self.classifier.train(X_train, y_train)
        
        # Step 4: Make predictions
        print("\n🔄 Step 4: Making predictions on test set...")
        y_pred = self.classifier.predict(X_test)
        
        # Step 5: Evaluate performance
        print("\n🔄 Step 5: Evaluating model performance...")
        self.evaluator.print_evaluation_summary(y_test, y_pred)
        
        # Step 6: Save the model
        print(f"\n💾 Saving model to {self.model_path}...")
        self.classifier.save_model(self.model_path)
        
        # Save evaluation report
        self.evaluator.save_report(y_test, y_pred, "evaluation_report.txt")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'metrics': self.evaluator.results.get('metrics', {})
        }
    
    def predict_image(self, image_path: str) -> dict:
        """
        Predict scratch status for a single image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Prediction result dictionary.
        """
        # Load model if not trained
        if not self.classifier.is_trained:
            if os.path.exists(self.model_path):
                self.classifier.load_model(self.model_path)
            else:
                raise RuntimeError(
                    f"No trained model found. Please train first or provide model at {self.model_path}"
                )
        
        # Make prediction
        result = self.classifier.predict_single_image(image_path)
        return result
    
    def batch_predict(self, image_paths: list) -> list:
        """
        Predict scratch status for multiple images.
        
        Args:
            image_paths: List of image file paths.
            
        Returns:
            List of prediction results.
        """
        results = []
        for img_path in image_paths:
            result = self.predict_image(img_path)
            result['image_path'] = img_path
            results.append(result)
        return results


def main():
    """Main entry point for the scratch detection pipeline."""
    parser = argparse.ArgumentParser(
        description="Scratch Detection Pipeline - Identify scratched text in documents"
    )
    parser.add_argument(
        '--predict', 
        type=str, 
        help='Path to image for prediction'
    )
    parser.add_argument(
        '--clean-dir', 
        type=str, 
        default='clean',
        help='Directory containing clean text images'
    )
    parser.add_argument(
        '--scratched-dir', 
        type=str, 
        default='scratched',
        help='Directory containing scratched text images'
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default='scratch_model.pkl',
        help='Path to save/load the model'
    )
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.3,
        help='Fraction of data for testing (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ScratchDetectionPipeline(model_path=args.model)
    
    if args.predict:
        # Prediction mode
        print(f"\n🔍 Predicting for: {args.predict}")
        result = pipeline.predict_image(args.predict)
        print("\n📋 Prediction Result:")
        print(f"   Image: {args.predict}")
        print(f"   Prediction: {result.get('prediction', 'N/A')}")
        print(f"   Confidence: {result.get('confidence', 'N/A')}")
    else:
        # Training mode
        image_paths, labels = pipeline.load_dataset(
            clean_dir=args.clean_dir,
            scratched_dir=args.scratched_dir
        )
        
        if len(image_paths) == 0:
            print("❌ No images found in the dataset directories!")
            return
        
        results = pipeline.train_and_evaluate(
            image_paths, 
            labels,
            test_size=args.test_size
        )
        
        print("\n✅ Pipeline completed successfully!")
        print(f"   Model saved to: {args.model}")
        print(f"   Report saved to: evaluation_report.txt")


if __name__ == "__main__":
    main()
