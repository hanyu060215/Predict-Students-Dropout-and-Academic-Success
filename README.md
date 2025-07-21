# Student Dropout and Academic Success Prediction

A comprehensive machine learning solution using Support Vector Machines (SVM) to predict student outcomes in higher education. This project classifies students into three categories: **Dropout**, **Graduate**, or **Enrolled**.

## 📊 Project Overview

Based on the research paper by M.V.Martins, D. Tolledo, J. Machado, L. M.T. Baptista, V.Realinho (2021) "Early prediction of student's performance in higher education: a case study", this project implements a robust machine learning pipeline to identify at-risk students and predict academic success.

### 🎯 Objectives
- Predict student outcomes early in their academic journey
- Help educational institutions identify at-risk students
- Enable proactive intervention strategies
- Achieve high prediction accuracy using optimized SVM models

## 🗂️ Dataset

The dataset contains **4,424 student records** with **36 features** covering:

### Categorical Features (18):
- **Demographics**: Marital status, Gender, Nationality, Age at enrollment
- **Academic Background**: Previous qualification, Course, Application mode/order
- **Attendance**: Daytime/evening attendance
- **Family Background**: Parents' qualifications and occupations
- **Financial Status**: Debtor status, Tuition fees, Scholarship holder
- **Special Circumstances**: Displaced, Educational special needs, International status

### Numerical Features (18):
- **Academic Performance**: Grades, curricular units (credited, enrolled, approved)
- **Evaluations**: 1st and 2nd semester performance metrics
- **Economic Indicators**: Unemployment rate, Inflation rate, GDP

### Target Variable:
- **Dropout** (0): Students who left without completing
- **Graduate** (1): Students who successfully completed their studies
- **Enrolled** (2): Students currently active in their programs

## 🛠️ Technical Implementation

### Machine Learning Pipeline:

1. **Data Preprocessing**
   - One-hot encoding for categorical variables
   - StandardScaler normalization for numerical features
   - RandomOverSampler for handling class imbalance
   - Train/Validation/Test split (80%/10%/10%)

2. **Model Development**
   - Multiple SVM kernel testing (Linear, RBF, Polynomial, Sigmoid)
   - Hyperparameter optimization using GridSearchCV
   - 5-fold cross-validation for robust evaluation

3. **Evaluation Metrics**
   - Accuracy and F1-score (weighted)
   - Detailed classification reports
   - Confusion matrix visualization
   - Class-wise performance analysis

### Key Functions:

#### Data Preprocessing
- `one_hot_encode_categorical()`: Converts categorical variables to binary features
- `scale_numerical()`: Standardizes numerical features using StandardScaler
- `oversample_data()`: Handles class imbalance using RandomOverSampler
- `preprocess_and_split()`: Complete preprocessing pipeline

#### Model Training & Evaluation
- `train_svm_with_different_kernels()`: Compares performance across kernel types
- `hyperparameter_tuning_svm()`: Optimizes hyperparameters using GridSearchCV
- `evaluate_svm_detailed()`: Comprehensive model evaluation with visualizations
- `compare_svm_models()`: Final model comparison on test set

## 📈 Results

### Model Performance (Validation Set):
| Kernel | F1-Score | Performance |
|--------|----------|-------------|
| **RBF** | **0.7665** | 🥇 Best |
| Linear | 0.7650 | 🥈 Very Close |
| Polynomial | 0.7650 | 🥉 Strong |
| Sigmoid | 0.6670 | ❌ Poor |

### Key Insights:
- **RBF kernel** achieved the best performance, indicating non-linear relationships in the data
- **Linear and polynomial** kernels performed nearly as well, suggesting mixed linear/non-linear patterns
- **Hyperparameter tuning** further improved the best model's performance
- The model successfully handles the **multi-class classification** challenge

### Hyperparameter Optimization:
- **C parameter** (regularization): [0.1, 1, 10, 100]
- **Gamma parameter**: ['scale', 'auto', 0.001, 0.01, 0.1, 1]
- **Degree parameter** (polynomial): [2, 3, 4]


### Expected Output:
- Dataset shape information
- Kernel comparison results
- Hyperparameter tuning progress
- Final model evaluation with metrics
- Confusion matrix visualization
- Class-wise performance breakdown


## 🔬 Research Background

This implementation is based on the research paper:
> M.V.Martins, D. Tolledo, J. Machado, L. M.T. Baptista, V.Realinho. (2021)  
> "Early prediction of student's performance in higher education: a case study"  
> *Trends and Applications in Information Systems and Technologies*, vol.1,  
> in Advances in Intelligent Systems and Computing series. Springer.  
> DOI: 10.1007/978-3-030-72657-7_16

## 🎓 Educational Impact

This project demonstrates:
- **Early Warning Systems**: Identify at-risk students before it's too late
- **Resource Allocation**: Help institutions focus support where needed most
- **Intervention Strategies**: Enable targeted academic and financial support
- **Data-Driven Decisions**: Support evidence-based educational policies

## 🔧 Technical Features

- **Robust Preprocessing**: Handles mixed data types with proper encoding and scaling
- **Class Imbalance Handling**: Uses oversampling to ensure fair representation
- **Cross-Validation**: Ensures reliable model performance estimates
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Fallback Compatibility**: Works with or without seaborn for visualizations
- **Modular Design**: Clean, reusable functions for each pipeline step


## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Note**: This project is for educational and research purposes. Always ensure compliance with data privacy regulations when working with student data.
