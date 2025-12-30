# Scratch Detection System

A machine learning–based system for detecting scratched/crossed-out words in scanned document images.

## Project Overview

This system automatically classifies whether a word or text region in a document image is **scratched/crossed out** or **clean**. It uses HOG (Histogram of Oriented Gradients) features with an SVM (Support Vector Machine) classifier.

## Architecture

The system follows a modular 4-agent architecture:

| Agent | Module | Responsibility |
|-------|--------|----------------|
| Agent 1 | `agent1_preprocessing.py` | Image preprocessing (grayscale, thresholding, resizing) |
| Agent 2 | `agent2_feature_extraction.py` | HOG feature extraction |
| Agent 3 | `agent3_classification.py` | SVM-based classification |
| Agent 4 | `agent4_evaluation.py` | Performance evaluation and metrics |

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install opencv-python numpy scikit-learn scikit-image matplotlib
```

## Dataset Structure

Organize your dataset as follows:
```
project/
├── clean/
│   ├── clean_0.png
│   ├── clean_1.png
│   └── ...
├── scratched/
│   ├── scratched_0.png
│   ├── scratched_1.png
│   └── ...
```

## Usage

### Train and Evaluate Model
```bash
python main.py
```

### Predict on a Single Image
```bash
python main.py --predict path/to/image.png
```

### Command Line Options
```
--clean-dir      Directory with clean images (default: clean)
--scratched-dir  Directory with scratched images (default: scratched)
--model          Path to save/load model (default: scratch_model.pkl)
--test-size      Test set fraction (default: 0.3)
--predict        Image path for single prediction
```

## Sample API Output

```json
{
  "prediction": "scratched",
  "confidence": 0.91,
  "label": 1
}
```

## Success Criteria

- Accuracy ≥ 85%
- Balanced Precision and Recall
- Fast inference time
- No dependency on pretrained models

## Files

| File | Description |
|------|-------------|
| `main.py` | Main pipeline script |
| `agent1_preprocessing.py` | Image preprocessing module |
| `agent2_feature_extraction.py` | HOG feature extraction module |
| `agent3_classification.py` | SVM classifier module |
| `agent4_evaluation.py` | Evaluation metrics module |
| `generate_dataset.py` | Synthetic dataset generator |

## Author

Siddhanth M

## References

1. Dalal, N., & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*
2. Scikit-learn Documentation
3. OpenCV Image Processing Tutorials
