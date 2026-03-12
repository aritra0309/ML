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
