# Emotion Classification using VGG16  

### 📌 Overview  
This project aims to classify human emotions from images using a **VGG16** model. The dataset consists of images categorized into seven different emotions:  

- **happy**  
- **neutral**  
- **disgust**  
- **angry**  
- **fear**  
- **disregard**  
- **surprise**  
- **sorrow**  

The model is optimized using **Optuna** to find the best hyperparameters and trained using **TensorFlow/Keras**. This method was developed during a **MachineHack Hackathon**, and the dataset was also provided by **MachineHack**.  

---

### 🛠 Steps Followed  

#### 🔹 Step 1 - Load Data  
The dataset used in this project was taken from a **MachineHack Hackathon**. It consists of images labeled with their respective emotions.  

#### 🔹 Step 2 - Preprocessing  
- Removed all corrupt images.  
- Resized all images to **(224, 224, 3)** to match the VGG16 input format.  
- Organized the images into subfolders based on their emotion labels, as provided in the training CSV file.  

#### 🔹 Step 3 - Data Splitting  
The dataset was split into **training (80%)** and **validation (20%)** sets using `ImageDataGenerator`.  

#### 🔹 Step 4 - Hyperparameter Optimization  
The **Optuna** library was used to optimize key parameters such as:  
- Learning rate  
- Number of dense units  
- Dropout rate  

#### 🔹 Step 5 - Model Training  
The best hyperparameters obtained from **Optuna** were used to train the model using:  
- **Adam optimizer**  
- **Categorical cross-entropy loss**  
- **Accuracy as the evaluation metric**  

---

### 🚀 Installation & Requirements  

To run this project, install the necessary dependencies:  

```bash
pip install tensorflow optuna numpy pandas matplotlib
```

---

### 📊 Results & Performance  
- The model achieved **high accuracy** in classifying emotions.  
- Using **Optuna**, the model's performance improved significantly by tuning hyperparameters.  
- Further improvements can be made by fine-tuning VGG16 layers and increasing dataset diversity.  

---

### 🎯 Future Scope  
- Implement **real-time emotion recognition** using OpenCV.  
- Experiment with **different architectures** like ResNet50, EfficientNet, etc.  
- Introduce **data augmentation** for better generalization.  

---

# Image Classification using CLIP and SVC

## Overview
This project utilizes OpenAI's CLIP model to extract feature embeddings from images and classifies them using a Support Vector Classifier (SVC). The classification process is optimized using Optuna for hyperparameter tuning. This method was also developed during a **MachineHack Hackathon**, and the dataset was sourced from **MachineHack**.

## Methodology
1. **Load Data**: The dataset is preloaded and consists of labeled images organized into folders.
2. **Preprocessing**: Images are resized, normalized, and processed using the CLIP model to obtain feature embeddings.
3. **Train-Test Splitting**: The extracted features are split into training and validation sets using an 80-20 ratio.
4. **Feature Scaling**: StandardScaler is applied to normalize feature vectors before training SVC.
5. **Hyperparameter Optimization**: Optuna is used to find the best values for:
   - `C` (regularization parameter)
   - `kernel` (linear, RBF, polynomial, or sigmoid)

## Advantages and Disadvantages
### ✅ Advantages:
- Lightweight model with reduced training time
- Faster optimization through Optuna
- Efficient feature extraction using CLIP

### ❌ Disadvantages:
- Encoding images using CLIP is slow
- Not practical for real-time applications due to encoding overhead

## Installation
Ensure you have Python installed along with the necessary dependencies. Install the required libraries using:

```bash
pip install torch torchvision PIL open_clip scikit-learn optuna numpy
```

## Usage
Run the script to extract features, train the SVC model, and optimize hyperparameters:

## 📌 Author  
Developed by **Abin John**  

---

This project was developed during a **MachineHack Hackathon**, and both methods were applied to classify emotions from images using **VGG16** and **CLIP+SVC**. The dataset was provided by **MachineHack**.

