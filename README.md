
---

# Cats vs Dogs Classification with SVM

## Project Overview
This project uses a **Support Vector Machine (SVM)** classifier to differentiate between images of cats and dogs. Feature extraction is performed using a pre-trained **VGG16** model, followed by training the SVM classifier on extracted features.

---

## Dataset
The dataset used for this project is the **Dogs vs Cats** dataset available on Kaggle. It contains images of cats and dogs distributed across two categories.  

- Dataset Link: [Dogs vs Cats - Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

### Steps to Use the Dataset:
1. Download the dataset from Kaggle.
2. Place the `kagglecatsanddogs_5340.zip` file in the specified path.
3. The script will automatically extract and process the images.

---

## Project Workflow

### 1. **Unzipping the Dataset**
The dataset zip file is extracted to a specified directory.

### 2. **Loading Images and Labels**
Images are loaded and preprocessed:
- **Resize**: Images are resized to `(150, 150)` pixels.
- **Preprocessing**: Pixel values are scaled using `VGG16` preprocessing.

Labels are assigned as follows:
- `Cat` = 0
- `Dog` = 1

### 3. **Splitting Data**
The dataset is split into training and testing sets with an 80:20 ratio.

### 4. **Feature Extraction**
Feature extraction is performed using the pre-trained **VGG16** model:
- **Base Model**: The convolutional base of VGG16 (without the top layers) extracts meaningful features.
- **Output Shape**: Features are flattened into a 2D array for input into the SVM.

### 5. **Training the SVM Classifier**
An SVM with the following parameters is trained:
- **Kernel**: RBF
- **C**: 1
- **Gamma**: 'scale'

### 6. **Model Evaluation**
The SVM classifier is evaluated on the test set using:
- **Accuracy Score**: Measures the overall accuracy of predictions.
- **Confusion Matrix**: Visualizes true vs. predicted labels.

### 7. **Model Saving and Visualization**
- The trained SVM model is saved as `svm_model.pkl`.
- A confusion matrix heatmap is displayed for better interpretation of results.

---

## Dependencies

Install the required libraries:
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn tqdm
```

---

## Running the Code

1. Place the dataset zip file (`kagglecatsanddogs_5340.zip`) in the path specified in the script.
2. Run the script:
   ```bash
   python classify_cats_dogs_svm.py
   ```
3. The script will:
   - Extract the dataset
   - Load and preprocess images
   - Extract features using VGG16
   - Train and evaluate the SVM classifier
4. Results, including accuracy and confusion matrix, will be displayed.

---

## Results

- **Test Accuracy**: Displays the percentage of correct predictions.
- **Confusion Matrix**: Shows the distribution of true vs. predicted labels.

Example Output:
```
Test Accuracy: 0.95
Confusion Matrix:
[[900  50]
 [ 40 910]]
```

---

## Notes

1. Ensure sufficient RAM/VRAM for feature extraction with VGG16, as it processes image batches.
2. Adjust the `target_size` and model parameters based on dataset quality or computational limits.
3. Use GPU for faster computation with TensorFlow.

---

