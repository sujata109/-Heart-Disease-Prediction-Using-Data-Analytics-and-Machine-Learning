# Heart Disease Prediction Using Data Analytics and Machine Learning

**Author:** Sujata Samanta

---

## Project Description

This project builds an end-to-end machine-learning pipeline to predict whether a patient is likely to have heart disease, based on 13 clinical features from the **Cleveland Heart Disease dataset**. The goal is to demonstrate how data analytics and supervised classification can assist early clinical screening.

The best-performing model — a **Support Vector Machine (SVC, rbf kernel, C=10, gamma=0.01)** wrapped in a scikit-learn `Pipeline` with `StandardScaler` — achieves **84.29 % mean 5-fold cross-validation accuracy** and **80.33 % held-out test accuracy**. The final pipeline is serialised to `heart_disease_pipeline.pkl` for deployment.

---

## Dataset

| Property | Value |
|---|---|
| **Name** | Cleveland Heart Disease Dataset |
| **Source** | [Kaggle — Heart Disease UCI](https://www.kaggle.com/code/farzadnekouei/heart-disease-prediction/input) |
| **File** | `heart.csv` |
| **Rows** | 303 |
| **Features** | 13 clinical + 1 target |
| **Target** | `1` = Heart Disease, `0` = No Heart Disease |
| **Class balance** | 165 positive (54.5 %), 138 negative (45.5 %) |

### Feature Descriptions

| Column | Description |
|---|---|
| `age` | Age of the patient (years) |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels coloured by fluoroscopy (0–4) |
| `thal` | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect) |
| `target` | **Target** — 1 = Heart Disease, 0 = No Heart Disease |

---

## Technologies Used

| Category | Library / Tool |
|---|---|
| Language | Python 3.9+ |
| Data manipulation | NumPy, Pandas |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | scikit-learn (SVC, LogisticRegression, RandomForestClassifier, Pipeline, StandardScaler, GridSearchCV, StratifiedKFold) |
| Model persistence | pickle (built-in) |
| Notebook environment | Jupyter Notebook / JupyterLab |

---

## Project Structure

```
├── heart.csv                        # Cleveland Heart Disease dataset
├── SujataSamanta_Heart Disease Prediction Using Data Analytics and Machine Learning.ipynb                # Main Jupyter notebook (EDA → training → evaluation → prediction)
├── heart_disease_pipeline.pkl       # Saved deployment pipeline (StandardScaler + SVC)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Setup & Run Instructions

### 1. Clone / download the repository

```bash
git clone <your-repo-url>
cd ibm-bubli
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook "SujataSamanta_Heart Disease Prediction Using Data Analytics and Machine Learning.ipynb"
```

Run all cells top-to-bottom using **Kernel → Restart & Run All**.

### 5. Use the saved pipeline for inference

```python
import pickle, pandas as pd

pipeline = pickle.load(open('heart_disease_pipeline.pkl', 'rb'))

features = ['age', 'sex', 'cp', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
sample = pd.DataFrame([[45, 1, 0, 147, 1, 0.0, 1, 3, 3]], columns=features)

print(pipeline.predict(sample))   # [0] → No Heart Disease  |  [1] → Heart Disease
```

---

## Model Performance Summary

| Model | Best CV Accuracy | Test Accuracy |
|---|---|---|
| **SVC (rbf, C=10, gamma=0.01)** | **84.29 %** | **80.33 %** |
| Logistic Regression (C=1) | 82.23 % | 78.69 % |
| Random Forest (n_estimators=20) | 81.83 % | 75.41 % |

- **Train / Test split:** 242 train / 61 test (stratified, `random_state=2`)
- **Cross-validation:** 5-fold `StratifiedKFold`
- **Feature selection:** 9 features selected — `age, sex, cp, thalach, exang, oldpeak, slope, ca, thal`
- **Deployed model:** `StandardScaler → SVC (rbf, C=10, gamma=0.01)`

---

## Key Findings

- **Chest pain type (`cp`)** and **thalassemia (`thal`)** are the strongest predictors of heart disease.
- Patients **with heart disease** tend to have **higher maximum heart rate (`thalach`)** and **lower ST depression (`oldpeak`)**.
- **5 features were dropped** after correlation analysis (`trestbps`, `chol`, `fbs`, `restecg`, `ca` partially — see notebook for full analysis).
- SVM with RBF kernel outperforms linear models on this dataset, suggesting **non-linear decision boundaries** are important.

---
