import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from typing import Dict, List, Optional


class ModelEvaluator:
    """Evaluates scratch detection model performance."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            class_names: Names for classes (default: ['clean', 'scratched']).
        """
        self.class_names = class_names or ['clean', 'scratched']
        self.results = {}
    
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            
        Returns:
            Dictionary of metric names and values.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_clean': precision_score(y_true, y_pred, pos_label=0, zero_division=0),
            'precision_scratched': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            'recall_clean': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            'recall_scratched': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            'f1_clean': f1_score(y_true, y_pred, pos_label=0, zero_division=0),
            'f1_scratched': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        self.results['metrics'] = metrics
        return metrics
    
    def generate_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        output_dict: bool = False
    ) -> str:
        """
        Generate a detailed classification report.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            output_dict: Whether to return as dictionary.
            
        Returns:
            Classification report as string or dictionary.
        """
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.class_names,
            output_dict=output_dict,
            zero_division=0
        )
        
        self.results['classification_report'] = report
        return report
    
    def generate_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            
        Returns:
            Confusion matrix as numpy array.
        """
        cm = confusion_matrix(y_true, y_pred)
        self.results['confusion_matrix'] = cm
        return cm
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot and optionally save confusion matrix.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            save_path: Path to save the plot (optional).
        """
        cm = self.generate_confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names
        )
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title('Scratch Detection - Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def cross_validate(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Trained classifier model.
            X: Feature matrix.
            y: Label array.
            cv: Number of cross-validation folds.
            
        Returns:
            Dictionary with cross-validation scores.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        cv_results = {
            'cv_scores': scores.tolist(),
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }
        
        self.results['cross_validation'] = cv_results
        return cv_results
    
    def print_evaluation_summary(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> None:
        """
        Print a comprehensive evaluation summary.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
        """
        print("=" * 60)
        print("SCRATCH DETECTION - EVALUATION REPORT")
        print("=" * 60)
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred)
        
        # Print accuracy
        print(f"\n📊 Overall Accuracy: {metrics['accuracy']:.2%}")
        
        # Print classification report
        print("\n" + "-" * 60)
        print("CLASSIFICATION REPORT")
        print("-" * 60)
        report = self.generate_classification_report(y_true, y_pred)
        print(report)
        
        # Print confusion matrix
        print("-" * 60)
        print("CONFUSION MATRIX")
        print("-" * 60)
        cm = self.generate_confusion_matrix(y_true, y_pred)
        print(f"\n                 Predicted")
        print(f"                 Clean  Scratched")
        print(f"Actual Clean     {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"Actual Scratched {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        print("\n" + "=" * 60)
        print("END OF REPORT")
        print("=" * 60)
    
    def save_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        filepath: str
    ) -> None:
        """
        Save evaluation report to a text file.
        
        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            filepath: Path to save the report.
        """
        metrics = self.compute_metrics(y_true, y_pred)
        report = self.generate_classification_report(y_true, y_pred)
        cm = self.generate_confusion_matrix(y_true, y_pred)
        
        with open(filepath, 'w') as f:
            f.write("SCRATCH DETECTION - EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.2%}\n\n")
            f.write("Classification Report:\n")
            f.write("-" * 50 + "\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write("-" * 50 + "\n")
            f.write(f"                 Predicted\n")
            f.write(f"                 Clean  Scratched\n")
            f.write(f"Actual Clean     {cm[0,0]:5d}  {cm[0,1]:5d}\n")
            f.write(f"Actual Scratched {cm[1,0]:5d}  {cm[1,1]:5d}\n")
        
        print(f"Report saved to {filepath}")


if __name__ == "__main__":
    # Demo with sample data
    np.random.seed(42)
    
    # Simulated predictions
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    
    evaluator = ModelEvaluator()
    evaluator.print_evaluation_summary(y_true, y_pred)
