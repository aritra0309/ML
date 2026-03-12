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

---
### Calorie Expenditure Prediction
**File:** `calorie-predictor-2.ipynb`  
**Dataset:** Kaggle Playground Series S5E5

This project predicts the number of **calories burned during physical activity** using physiological and activity-related features.

**Pipeline**
- Dataset loaded from Kaggle Playground Series
- Categorical feature (`Sex`) encoded using `LabelEncoder`
- Extensive **feature engineering** using cross-feature interactions:
  - Multiplication
  - Addition
  - Division between numerical variables
- Target variable (`Calories`) transformed using **log1p** to stabilize variance
- Data split into **training and validation sets**

**Model**
- **XGBoost Regressor**
- GPU training enabled (`gpu_hist`)
- Hyperparameters optimized using **RandomizedSearchCV**

**Evaluation**
- Metric: **Mean Squared Log Error (MSLE)**
- Best CV MSLE ≈ **0.00030**
- Validation MSLE ≈ **0.0036**

Final predictions are transformed back using `expm1` and exported as a **Kaggle submission file**.

---
### Drawing with Vision-Language Models (PaLI-Gemma)

**File:** `drawing-with-llm-pali-gemma-2.ipynb`  
**Competition:** Kaggle – Drawing with LLMs

This project explores **multimodal generative models** to create SVG illustrations from text prompts and evaluate them using automated visual reasoning metrics.

**Pipeline**

1. **Image Generation**
   - Uses **Stable Diffusion** to generate bitmap images from text prompts.
   - Prompts are structured with prefixes, suffixes, and negative prompts to control style.

2. **Bitmap → SVG Conversion**
   - Generated images are converted to **vector graphics (SVG)**.
   - Uses:
     - Superpixel segmentation (SLIC)
     - Color quantization
     - Polygon and ellipse fitting
     - Edge and saliency-based feature extraction

3. **Model Evaluation**

Generated SVGs are scored using a **custom evaluation pipeline** consisting of:

- **PaLI-Gemma Vision-Language Model**
  - Performs visual question answering (VQA) to evaluate semantic correctness.

- **CLIP-based Aesthetic Model**
  - Measures visual quality of generated images.

- **OCR Robustness**
  - Tests image stability under compression and transformations.

4. **Final Metric**

The final score combines:

- VQA score  
- Aesthetic score  
- Complexity penalty for overly large SVG files  

using a **harmonic mean–based scoring function**.

**Additional Features**

- GPU optimization with **4-bit quantization (bitsandbytes)**
- Automatic **CUDA memory management**
- Multi-attempt generation with **best-result selection**
- Visualization of generated bitmap vs SVG outputs
- Performance tracking for large-scale inference

This project demonstrates experimentation with **multimodal AI, diffusion models, and vector graphic generation pipelines**.
