"""
CodeAlpha Data Science Internship - Task 1: Iris Flower Classification
Author: CodeAlpha Intern
Description: End-to-end Machine Learning pipeline to classify Iris flower species
             (Iris-setosa, Iris-versicolor, Iris-virginica) using Sepal and Petal dimensions.
"""

import os
import sys
import json
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore deprecation and user warnings for clean console outputs
warnings.filterwarnings('ignore')

# Ensure UTF-8 output encoding if possible
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Machine Learning Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Set plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_palette("Set2")


def print_banner(title: str):
    """Prints a styled terminal banner."""
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title.center(66)}")
    print(f"{line}\n")


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads the Iris dataset from a CSV file.
    Drops unnecessary columns like 'Id' if present.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    
    # Normalize column names if needed
    column_mapping = {
        'SepalLengthCm': 'sepal_length',
        'SepalWidthCm': 'sepal_width',
        'PetalLengthCm': 'petal_length',
        'PetalWidthCm': 'petal_width',
        'Species': 'species'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    return df


def perform_eda(df: pd.DataFrame, output_dir: str):
    """
    Performs Exploratory Data Analysis and saves informative visualizations.
    """
    print_banner("1. EXPLORATORY DATA ANALYSIS (EDA)")
    
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 Records:")
    print(df.head().to_string())
    
    print("\nDataset Information & Summary Statistics:")
    print(df.describe().round(2).to_string())
    
    print("\nMissing Values Count:")
    print(df.isnull().sum().to_string())
    
    print("\nTarget Class Distribution:")
    print(df['species'].value_counts().to_string())
    
    os.makedirs(output_dir, exist_ok=True)
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    # 1. Feature Distributions (Histograms + KDE)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Feature Distributions across Iris Species', fontsize=16, fontweight='bold', y=0.98)
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx // 2, idx % 2]
        sns.histplot(data=df, x=feature, hue='species', kde=True, ax=ax, alpha=0.5, element="step")
        ax.set_title(f'Distribution of {feature.replace("_", " ").title()} (cm)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Length / Width (cm)')
        ax.set_ylabel('Count')
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'feature_distributions.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {dist_path}")

    # 2. Pairplot
    pairplot = sns.pairplot(df, hue='species', diag_kind='kde', corner=False, markers=["o", "s", "D"])
    pairplot.fig.subplots_adjust(top=0.94)
    pairplot.fig.suptitle('Iris Dataset Pairplot - Feature Relationships by Species', fontsize=14, fontweight='bold')
    pairplot_path = os.path.join(output_dir, 'pairplot.png')
    pairplot.savefig(pairplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {pairplot_path}")

    # 3. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    numeric_df = df[feature_cols]
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.3f', cbar=True, square=True, linewidths=1)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=12)
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {heatmap_path}")

    # 4. Boxplots & Strip plots
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Box & Strip Plots by Species', fontsize=16, fontweight='bold', y=0.98)
    for idx, feature in enumerate(feature_cols):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(data=df, x='species', y=feature, hue='species', legend=False, ax=ax, width=0.4, palette="Set2", boxprops=dict(alpha=0.7))
        sns.stripplot(data=df, x='species', y=feature, ax=ax, color='black', alpha=0.4, jitter=0.2, size=4)
        ax.set_title(f'{feature.replace("_", " ").title()} by Species', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('cm')
    plt.tight_layout()
    boxplot_path = os.path.join(output_dir, 'feature_boxplots.png')
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {boxplot_path}")


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splits features and target, performs stratified train-test split, and scales features.
    """
    feature_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[feature_cols].values
    y = df['species'].values
    class_names = np.unique(y).tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, class_names, feature_cols


def train_and_compare_models(
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: list,
    output_dir: str
) -> dict:
    """
    Trains multiple Machine Learning models, computes evaluation metrics and cross-validation scores,
    and visualizes model performance.
    """
    print_banner("2. MODEL TRAINING & PERFORMANCE BENCHMARKING")
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200, random_state=42),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine (SVM)": SVC(kernel='rbf', probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Cross validation on training set
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted')
        recall = recall_score(y_test, y_test_pred, average='weighted')
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        cm = confusion_matrix(y_test, y_test_pred, labels=class_names)
        report = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True)

        results[name] = {
            'model': model,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }

    # Display comparison table in console
    comparison_df = pd.DataFrame([
        {
            'Algorithm': name,
            'Train Acc (%)': f"{res['train_accuracy'] * 100:.2f}%",
            'Test Acc (%)': f"{res['test_accuracy'] * 100:.2f}%",
            '5-Fold CV Mean (%)': f"{res['cv_mean'] * 100:.2f}% (+/-{res['cv_std'] * 100:.2f}%)",
            'Precision (%)': f"{res['precision'] * 100:.2f}%",
            'Recall (%)': f"{res['recall'] * 100:.2f}%",
            'F1 Score (%)': f"{res['f1_score'] * 100:.2f}%"
        }
        for name, res in results.items()
    ])
    
    print(comparison_df.to_string(index=False))

    # Detailed Classification Reports
    print("\nDetailed Test Set Classification Reports:")
    for name, res in results.items():
        print(f"\n--- {name} ---")
        print(classification_report(y_test, res['model'].predict(X_test_scaled), target_names=class_names))

    # Visualizations: Model Comparison Bar Chart
    plt.figure(figsize=(11, 6))
    x = np.arange(len(results))
    width = 0.35
    
    acc_scores = [res['test_accuracy'] * 100 for res in results.values()]
    f1_scores = [res['f1_score'] * 100 for res in results.values()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, acc_scores, width, label='Test Accuracy (%)', color='#4C72B0')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score (%)', color='#55A868')
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Machine Learning Algorithm Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(list(results.keys()), rotation=15, ha='right', fontsize=10, fontweight='bold')
    ax.set_ylim(80, 105)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # Add data labels
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Saved: {comparison_path}")

    # Confusion Matrix Multi-plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
        ax.set_title(f"{name}\nAcc: {res['test_accuracy'] * 100:.1f}%", fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Species', fontsize=10)
        ax.set_ylabel('Actual Species', fontsize=10)
    
    # Hide unused subplot
    axes[5].axis('off')
    plt.suptitle('Confusion Matrices on Test Data (n=30)', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {cm_path}")

    return results


def tune_and_save_best_model(
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    class_names: list,
    models_dir: str
):
    """
    Performs hyperparameter optimization on SVM / Random Forest, evaluates on test set,
    and serializes the production model & scaler.
    """
    print_banner("3. HYPERPARAMETER TUNING & MODEL ARTIFACT EXPORT")
    
    # SVM Parameter Grid
    param_grid = {
        'C': [0.1, 1.0, 10.0, 50.0],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1.0],
        'kernel': ['rbf', 'linear', 'poly']
    }
    
    grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"Best Parameters Found: {grid.best_params_}")
    print(f"Best 5-Fold CV Accuracy: {grid.best_score_ * 100:.2f}%")
    print(f"Tuned Model Test Accuracy: {test_score * 100:.2f}%")

    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'best_iris_model.pkl')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    metadata_path = os.path.join(models_dir, 'model_metadata.json')

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        'model_type': best_model.__class__.__name__,
        'best_parameters': grid.best_params_,
        'train_cv_accuracy': float(grid.best_score_),
        'test_accuracy': float(test_score),
        'class_names': class_names,
        'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[OK] Serialized Model: {model_path}")
    print(f"[OK] Serialized Scaler: {scaler_path}")
    print(f"[OK] Model Metadata: {metadata_path}")

    return best_model, metadata


def predict_sample(model, scaler, measurements: list, class_names: list):
    """
    Takes 4 raw measurements [sepal_length, sepal_width, petal_length, petal_width],
    scales them, and outputs predicted species with probability distribution.
    """
    arr = np.array(measurements).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    prediction = model.predict(arr_scaled)[0]
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(arr_scaled)[0]
        prob_dict = {cls: round(float(prob) * 100, 2) for cls, prob in zip(class_names, probabilities)}
    else:
        prob_dict = {}

    return prediction, prob_dict


def interactive_predict_mode(model, scaler, class_names):
    """
    Runs an interactive terminal loop allowing users to test custom flower measurements.
    """
    print_banner("4. INTERACTIVE FLOWER PREDICTION MODE")
    print("Enter flower measurements in centimeters to predict species.")
    print("Type 'q' or 'exit' anytime to quit.\n")

    sample_benchmarks = [
        ("Typical Setosa", [5.1, 3.5, 1.4, 0.2]),
        ("Typical Versicolor", [6.0, 2.7, 5.1, 1.6]),
        ("Typical Virginica", [6.5, 3.0, 5.5, 1.8]),
    ]

    print("Sample Reference Values for Testing:")
    for name, vals in sample_benchmarks:
        pred, probs = predict_sample(model, scaler, vals, class_names)
        print(f"  * {name} {vals} -> Predicted: {pred} (Confidence: {probs.get(pred, 'N/A')}%)")
    print("-" * 70)

    while True:
        try:
            user_input = input("\nEnter [SepalLength, SepalWidth, PetalLength, PetalWidth] (e.g. 5.8, 2.7, 4.1, 1.0): ")
            if user_input.strip().lower() in ['q', 'exit', 'quit']:
                print("\nExiting interactive prediction mode.")
                break
            
            parts = [float(x.strip()) for x in user_input.split(',') if x.strip()]
            if len(parts) != 4:
                print("Error: Please provide exactly 4 comma-separated numerical values.")
                continue

            pred, probs = predict_sample(model, scaler, parts, class_names)
            
            print("\n" + "=" * 45)
            print(f"PREDICTED SPECIES: {pred}")
            print("=" * 45)
            print("Confidence Probabilities:")
            for cls, prob in probs.items():
                bar = "#" * int(prob // 5)
                print(f"  * {cls:<16}: {prob:>6.2f}% | {bar}")
            print("=" * 45)
        except ValueError:
            print("Invalid input! Please enter numbers only (e.g. 5.1, 3.5, 1.4, 0.2).")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="CodeAlpha Iris Flower Classification Machine Learning Pipeline")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI prediction mode")
    parser.add_argument("--predict", nargs=4, type=float, metavar=('SL', 'SW', 'PL', 'PW'), help="Predict species for given measurements: SepalLength SepalWidth PetalLength PetalWidth")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', 'iris.csv')
    figures_dir = os.path.join(base_dir, 'outputs', 'figures')
    models_dir = os.path.join(base_dir, 'outputs', 'models')

    if args.predict:
        model_path = os.path.join(models_dir, 'best_iris_model.pkl')
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        metadata_path = os.path.join(models_dir, 'model_metadata.json')
        if not os.path.exists(model_path):
            print("Model not found. Running training pipeline first...")
            df = load_dataset(data_path)
            X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, class_names, _ = prepare_data(df)
            train_and_compare_models(X_train_scaled, X_test_scaled, y_train, y_test, class_names, figures_dir)
            model, _ = tune_and_save_best_model(X_train_scaled, y_train, X_test_scaled, y_test, scaler, class_names, models_dir)
        else:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            with open(metadata_path, 'r') as f:
                class_names = json.load(f)['class_names']
        
        pred, probs = predict_sample(model, scaler, args.predict, class_names)
        print(f"\nMeasurements: {args.predict}")
        print(f"Predicted Species: {pred}")
        print("Confidence Breakdown:")
        for cls, prob in probs.items():
            print(f"  * {cls}: {prob:.2f}%")
        return

    print_banner("CODEALPHA DATA SCIENCE INTERNSHIP - TASK 1")
    print("Project: Iris Flower Classification Pipeline")
    print(f"Working Directory: {base_dir}")

    # Step 1: Load Data & Perform EDA
    df = load_dataset(data_path)
    perform_eda(df, figures_dir)

    # Step 2: Data Preprocessing & Splitting
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, class_names, feature_cols = prepare_data(df)

    # Step 3: Train & Benchmark Multiple Models
    results = train_and_compare_models(X_train_scaled, X_test_scaled, y_train, y_test, class_names, figures_dir)

    # Step 4: Hyperparameter Optimization & Artifact Saving
    best_model, metadata = tune_and_save_best_model(
        X_train_scaled, y_train, X_test_scaled, y_test, scaler, class_names, models_dir
    )

    print_banner("EXECUTION & EVALUATION COMPLETED SUCCESSFULLY")
    print(f"All figures saved to: {figures_dir}")
    print(f"All models saved to:  {models_dir}")

    # Run automated benchmark demonstrations
    print("\nAutomated Validation Benchmark:")
    test_samples = [
        ("Iris-setosa Benchmark", [5.0, 3.6, 1.4, 0.2]),
        ("Iris-versicolor Benchmark", [5.9, 3.0, 4.2, 1.5]),
        ("Iris-virginica Benchmark", [6.9, 3.1, 5.4, 2.1])
    ]
    for label, feats in test_samples:
        p, probs = predict_sample(best_model, scaler, feats, class_names)
        print(f"  * Input: {feats} ({label}) -> Predicted: {p} (Confidence: {probs[p]:.2f}%)")

    if args.interactive:
        interactive_predict_mode(best_model, scaler, class_names)


if __name__ == '__main__':
    main()
