# Patient Readmission Prediction

## Overview
This project predicts **hospital patient readmission risk** within 30 days using **Python, SQL, and Machine Learning**.  
The dataset consists of **120k+ hospital records** with demographic, treatment, and diagnostic information.  

The goal is to help hospitals identify **high-risk patients** and reduce preventable readmissions.  

---

## Tech Stack
- **Python** → Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib  
- **SQL (SQLite)** → for data storage and querying  
- **Imbalanced-learn (SMOTE)** → for handling class imbalance  
- **Jupyter Notebook** → for experimentation & visualization  

---

## Dataset
- **Source**: [Diabetes 130 US Hospitals Dataset (UCI Repository)](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008)  
- **Records**: ~120,000 patients  
- **Features**: 50+ (age, time in hospital, medications, comorbidities, etc.)  
- **Target**: `readmitted` → (1 = readmitted within 30 days, 0 = not readmitted)  

---

## Approach

### 1. Data Loading & Storage  
- Loaded raw dataset (`diabetic_data.csv`) into **SQLite** for structured querying.  

### 2. Data Cleaning  
- Removed patients who died during admission  
- Handled missing values  
- Converted target (`readmitted`) into binary classification (0/1)  

### 3. Exploratory Data Analysis  
- Visualized **class imbalance**
- ![Class Imbalance](Figure_1.png)

### 4. Feature Engineering  
- One-hot encoded categorical features (age groups)  
- Standardized numerical variables
- ![Feature Importance](Figure_2.png)


### 5. Modeling  
- Logistic Regression with **class balancing**  
- Applied **SMOTE oversampling** to handle minority class  

### 6. Evaluation  
- **Accuracy**: ~52% (after balancing)  
- **Recall (class 1)**: ~57% (significant improvement vs baseline)  
- Identified **top feature importance**  

---

## Key Insights
- Dataset is **highly imbalanced** (majority patients not readmitted).  
- **Top predictive factors**:  
  - Age group **20–30** (higher readmission risk)  
  - **Number of diagnoses**  
  - Elderly patients (**80–90**) also at higher risk  
- Balancing techniques (**SMOTE, class weights**) improve detection of at-risk patients.  

