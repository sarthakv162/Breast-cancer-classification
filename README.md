# Breast Cancer Detection Using Deep Learning

This repository presents a complete deep learning pipeline for detecting breast cancer from histopathological images using the BreaKHis dataset. The project uses TensorFlow/Keras and covers all stages from data preprocessing and visualization to model training and evaluation.

---

## Objective

To build a robust deep learning model capable of classifying breast cancer histopathology images into benign and malignant categories using convolutional neural networks (CNN).

---

## Dataset Overview

* **Name:** BreaKHis (Breast Cancer Histopathological Database)
* **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)
* **Image Format:** PNG
* **Magnifications:** 40X, 100X, 200X, 400X
* **Total Images:** 7,909

### Class Distribution

* **Benign Tumors:**

  * Adenosis (A)
  * Fibroadenoma (F)
  * Phyllodes Tumor (PT)
  * Tubular Adenoma (TA)

* **Malignant Tumors:**

  * Ductal Carcinoma (DC)
  * Lobular Carcinoma (LC)
  * Mucinous Carcinoma (MC)
  * Papillary Carcinoma (PC)

---

## Project Structure

```
├── model.ipynb               # Main Jupyter Notebook
├── dataset/                  # Dataset directory (downloaded manually)
├── results/                  # Plots, evaluation metrics
├── requirements.txt          # Python dependencies
└── README.md                 # Project description
```

---

## Installation

### Prerequisites

* Python 3.7+

### Installation Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/sarthakv162/Breast-cancer-classification.git
   cd Breast-cancer-classification
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis) and place it inside the `dataset/` folder.

---

## Methodology

### 1. Data Loading

* Uses `glob` and `pathlib` to recursively load all PNG images
* Parses image labels based on filename patterns

### 2. Visualization

* Side-by-side image grids of benign and malignant samples
* Subtype distribution plots

### 3. Data Preprocessing

* Resize images (e.g., 224x224)
* Normalize pixel values
* Encode labels (binary or categorical)

### 4. Model Architecture

* CNN with multiple Conv2D, MaxPooling2D, Dropout, and Dense layers
* Potential for transfer learning (e.g., EfficientNet, MobileNet)

### 5. Training

* Binary classification with `binary_crossentropy`
* Optimizer: Adam
* EarlyStopping, ReduceLROnPlateau
* Stratified train-test split

### 6. Evaluation

* Accuracy, Precision, Recall, F1-Score
* Confusion Matrix

---

## Results

Tumor classification Model

| Metric    | Value           |
| --------- | --------------- |                                        
| Accuracy  | 98.40%          |
| Precision | 97.90%          |
| Recall    | 99.25%          |
| F1-Score  | 98.57%          |

Benign Subtypes Classification

| Metric    | Value           |
| --------- | --------------- |
| Accuracy  | 91.47%          |
| Precision | 91.95%          |
| Recall    | 89.43%          |
| F1-Score  | 90.47%          |

Malignant Subtypes Classification

| Metric    | Value           |
| --------- | --------------- |
| Accuracy  | 92.12%          |
| Precision | 92.41%          |
| Recall    | 91.64%          |
| F1-Score  | 92.00%          |


---

## Future Improvements

* Add support for each magnification level separately
* Ensemble CNN architectures
* Integrate Grad-CAM for interpretability
* Quantize and export model to TensorFlow Lite or ONNX

---

## License

This repository is licensed under the MIT License.

---

## Acknowledgements

* Dataset: BreaKHis via Kaggle
* Libraries: TensorFlow, Keras, NumPy, OpenCV, Matplotlib

---

## Contributions

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repo, submit a pull request, or open an issue.

---

## Contact

Created by \Sarthak Verma — feel free to connect via [LinkedIn](https://www.linkedin.com/in/sarthak-verma-6002001b4/) or raise an issue for queries.
