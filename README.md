# Diabetic Retinopathy Detection using Transfer Learning & CNNs  
*A Deep Learning Project for Retinal Fundus Image Classification*

## 🔍 Project Overview  
This repository implements a deep-learning pipeline for the detection and classification of Diabetic Retinopathy (DR) from retinal fundus images, using convolutional neural networks (CNNs) with transfer-learning approaches. The model is trained on your dataset from Kaggle and covers preprocessing, model fine-tuning, evaluation, and classification into DR severity classes.

##  Key Features  
- Use of pre-trained CNN architectures (e.g., ResNet, Inception) for feature extraction and fine-tuning.  
- Data preprocessing including image resizing, augmentation, and class-balancing.  
- Multi-class classification of DR severity (for example: No DR, Mild, Moderate, Severe, Proliferative).  
- Model evaluation including accuracy, confusion matrix, and metrics.  
- Jupyter notebooks for experimentation and reproducibility.

## 📁 Dataset  
The dataset used for this project was published on Kaggle by the author. You can download / access it here:  
[Diabetic Retinopathy Data – Kaggle](https://www.kaggle.com/datasets/pramod036/diabetic-retinopathy-data)  


## 🧰 Environment Setup  
### Prerequisites  
- Python 3.8 or higher  
- GPU support (preferred for training)  
- Virtual environment recommended  

### Installation  
```bash
# Clone the repository
git clone https://github.com/pramodlv007/DiabeticRetinopathy-DetectionUsingTransferLearning-CNNs-
cd DiabeticRetinopathy-DetectionUsingTransferLearning-CNNs-

# Create and activate virtual env (example with venv)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

### Configuration

In my notebook / config file, set the path to the dataset:

```
DATA_DIR = "/path/to/kaggle/download/diabetic-retinopathy-data"
```

### Running the Notebooks / Experiments

* Launch Jupyter Notebook or open in VS Code.
* Load the dataset, preprocess images (e.g., resizing to 224×224), apply augmentation.
* Set up the model (transfer learning base + custom classification head).
* Train the model, validate, evaluate.
* Save the trained model checkpoint for inference.

##  Methodology

1. **Data Loading & Preprocessing**

   * Read images and labels from the dataset directory.
   * Resize images (commonly to 224×224), normalize pixel values, apply augmentations (flip, rotate, brightness).
   * Deal with class imbalance (e.g., by oversampling minority classes or using weighted loss).

2. **Transfer Learning Model Setup**

   * Choose base network (e.g., ResNet50 pretrained on ImageNet).
   * Replace top layers with classification head for DR severity categories.
   * Initially freeze base layers; fine-tune deeper layers after initial epochs.

3. **Training & Evaluation**

   * Use loss like categorical cross-entropy, optimizer like Adam.
   * Monitor validation performance, use callbacks (early stopping, checkpoints).
   * Evaluate model with metrics (accuracy, precision/recall per class, confusion matrix).

4. **Inference & Deployment (Optional)**

   * Load the trained model for single image inference.
   * Create a simple web interface or REST API to upload a fundus image and get DR classification.

