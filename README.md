### Introvert–Extrovert Personality Classification
**File:** `Introvert-Extrovert.ipynb`

This project predicts whether a person is an **Introvert or Extrovert** using behavioral and social activity features from the Kaggle Playground Series dataset.

**Pipeline:**
- Missing values handled using `SimpleImputer`
  - Mean imputation for numerical features  
  - Most frequent imputation for categorical features
- Categorical variables encoded using `LabelEncoder`
- Dataset split into **80% training and 20% validation**

**Models Used:**
- XGBoost Classifier  
- CatBoost Classifier  
- LightGBM Classifier  

**Ensemble Method:**
Predictions from the three models are combined by **averaging their probability outputs**, and the class with the highest probability is selected as the final prediction.

**Evaluation Metric:**  
Accuracy score on the validation set.

The trained ensemble is then used to generate predictions for the Kaggle test dataset.

### Pneumonia Detection using CNN
**File:** `Pneumonia_Prediction_CNN.ipynb`  
**Dataset:** Chest X-Ray Pneumonia Dataset (Kaggle)

This project builds a **Convolutional Neural Network (CNN)** to classify chest X-ray images as **Pneumonia or Normal**.

**Pipeline**
- Chest X-ray images loaded using OpenCV and resized to **150×150**
- Images converted to grayscale and normalized
- Data augmentation applied (random flips and brightness changes)
- Dataset split into **train, validation, and test sets**
- Class imbalance handled using **class weights**

**Model Architecture**
- Multiple **Conv2D layers** with ReLU activation
- **Batch Normalization** for stable training
- **MaxPooling layers** for spatial reduction
- **Dropout layers** for regularization
- Fully connected dense layer followed by **sigmoid output**

**Training Techniques**
- Early stopping
- Learning rate reduction
- Model checkpointing

**Performance**
- Test Accuracy: **~87%**
- Evaluated using:
  - Classification report
  - Confusion matrix
  - Prediction visualization on sample X-ray images
