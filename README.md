# CodeAlpha Data Science Internship — Task 1
# 🌸 Iris Flower Classification

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Library-Scikit--Learn%20%7C%20Pandas%20%7C%20Seaborn-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)]()

---

## 📌 Project Overview
This repository contains the complete implementation for **Task 1: Iris Flower Classification** developed as part of the **CodeAlpha Data Science Internship program**.

The goal of this project is to build an end-to-end Machine Learning pipeline that accurately classifies Iris flower samples into one of three species:
- 🌸 **Iris setosa**
- 🌸 **Iris versicolor**
- 🌸 **Iris virginica**

Classification is performed based on four botanical morphological features:
1. **Sepal Length** (in cm)
2. **Sepal Width** (in cm)
3. **Petal Length** (in cm)
4. **Petal Width** (in cm)

---

## 📂 Repository Structure

```
CodeAlpha_IrisFlowerClassification/
├── data/
│   └── iris.csv                          # Standard 150-sample Iris dataset
├── outputs/
│   ├── figures/                          # High-resolution generated plots & charts
│   │   ├── feature_distributions.png     # Histograms & KDE distribution per feature
│   │   ├── pairplot.png                  # Pairwise feature relationships by species
│   │   ├── correlation_heatmap.png       # Correlation matrix of numerical features
│   │   ├── feature_boxplots.png          # Box & strip plots for outlier inspection
│   │   ├── model_comparison.png          # Accuracy & F1-score comparison bar chart
│   │   └── confusion_matrices.png        # Multi-model confusion matrix grid
│   └── models/                           # Serialized production models & artifacts
│       ├── best_iris_model.pkl           # Tuned classifier artifact (SVM / Scikit-learn)
│       ├── scaler.pkl                    # Fitted StandardScaler object
│       └── model_metadata.json           # Model configuration, params, and metrics
├── iris_classification.py                # Standalone production pipeline & CLI tool
├── Iris_Flower_Classification.ipynb      # Step-by-step interactive Jupyter Notebook
├── requirements.txt                      # Project dependencies
└── README.md                             # Comprehensive project documentation
```

---

## 🔬 Dataset Exploration & Insights

- **Total Samples:** 150 instances (50 samples per class — perfectly balanced).
- **Missing Values:** 0 missing values across all features.
- **Key Findings:**
  - **Petal Length & Petal Width** exhibit an exceptionally high positive correlation ($r > 0.96$) and provide the strongest linear separability for *Iris-setosa*.
  - *Iris-setosa* is distinctly separated from the other two species with no overlap.
  - *Iris-versicolor* and *Iris-virginica* have slight boundary overlap in sepal dimensions, making non-linear classifiers (e.g. SVM with RBF/linear kernel, Decision Trees, and Random Forests) highly effective.

---

## 📊 Machine Learning Model Benchmarking

We benchmarked 5 industry-standard classification algorithms using **Stratified 80/20 Train-Test Split** and **5-Fold Stratified Cross-Validation**:

| Algorithm | Train Acc (%) | Test Acc (%) | 5-Fold CV Mean (%) | Precision (%) | Recall (%) | F1 Score (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM)** | **97.50%** | **96.67%** | **96.67% (±1.67%)** | **96.97%** | **96.67%** | **96.66%** |
| **Decision Tree Classifier** | 98.33% | 96.67% | 96.67% (±3.12%) | 96.97% | 96.67% | 96.66% |
| **Logistic Regression** | 95.83% | 93.33% | 95.83% (±2.64%) | 93.33% | 93.33% | 93.33% |
| **K-Nearest Neighbors (KNN)** | 97.50% | 93.33% | 95.83% (±2.64%) | 94.44% | 93.33% | 93.27% |
| **Random Forest Classifier** | 100.00% | 90.00% | 95.00% (±3.12%) | 90.24% | 90.00% | 89.97% |

### ⚙️ Hyperparameter Optimization
Using `GridSearchCV` across kernel types, regularization parameters ($C$), and gamma coefficients:
- **Optimal Hyperparameters:** `{'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}`
- **Best 5-Fold Cross-Validation Score:** **97.50%**
- **Model Artifacts:** Automatically saved to `outputs/models/best_iris_model.pkl`.

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Clone this repository and navigate into the project directory:

```bash
git clone https://github.com/<your-username>/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification
```

Create a virtual environment and install dependencies:

```bash
# Using standard Python venv
python -m venv .venv
# Activate on Windows:
.\.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 💻 How to Run

### Option 1: Run the End-to-End Pipeline
Executes data preprocessing, EDA, multi-model training, evaluation, plot generation, and automated validation tests:

```bash
python iris_classification.py
```

### Option 2: Predict Flower Species via Single CLI Command
Provide 4 values: `SepalLength SepalWidth PetalLength PetalWidth`:

```bash
python iris_classification.py --predict 5.1 3.5 1.4 0.2
```

**Output:**
```
Measurements: [5.1, 3.5, 1.4, 0.2]
Predicted Species: Iris-setosa
Confidence Breakdown:
  * Iris-setosa: 97.40%
  * Iris-versicolor: 1.59%
  * Iris-virginica: 1.01%
```

### Option 3: Launch Interactive Terminal Mode
```bash
python iris_classification.py --interactive
```

### Option 4: Open the Jupyter Notebook
```bash
jupyter lab Iris_Flower_Classification.ipynb
```

---

## 📈 Visualizations Showcase

All generated figures are saved in `outputs/figures/`:
1. `feature_distributions.png` — Distribution of each botanical dimension per species.
2. `pairplot.png` — Pairwise scatter and density relationships showing clustering.
3. `correlation_heatmap.png` — Correlation matrix between petal and sepal features.
4. `feature_boxplots.png` — Outlier analysis and statistical spread.
5. `model_comparison.png` — Model accuracy & F1-score comparison chart.
6. `confusion_matrices.png` — Confusion matrices across all evaluated models.

---

## 📄 Internship Submission Details & LinkedIn Post

- **Domain:** Data Science
- **Organization:** CodeAlpha ([www.codealpha.tech](https://www.codealpha.tech))
- **Task:** Task 1 — Iris Flower Classification
- **GitHub Repository Name:** `CodeAlpha_IrisFlowerClassification`

### Suggested LinkedIn Post Template:
```text
Excited to share that I have completed Task 1: Iris Flower Classification as part of my Data Science Internship with @CodeAlpha! 🚀

🌸 In this project, I developed an end-to-end Machine Learning pipeline to classify Iris flower species based on sepal and petal dimensions.

Key Highlights:
✅ Exploratory Data Analysis (EDA) with pairwise distributions & correlation analysis
✅ Implemented & benchmarked 5 ML algorithms (Logistic Regression, KNN, SVM, Decision Tree, Random Forest)
✅ Achieved 97.5% Cross-Validation Accuracy with hyperparameter-tuned Support Vector Machine (SVM)
✅ Built an interactive CLI prediction interface and exported production model artifacts

🔗 GitHub Repository: https://github.com/<your-username>/CodeAlpha_IrisFlowerClassification

Thank you @CodeAlpha for this wonderful learning opportunity! 🎓✨

#CodeAlpha #DataScience #MachineLearning #Python #ScikitLearn #AI #Internship #ArtificialIntelligence
```

---

## 📜 License
This project is open-source under the [MIT License](LICENSE).
