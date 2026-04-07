# Supervised Learning Classification: Vehicle Acceptability Prediction

A comprehensive machine learning classification system for predicting vehicle acceptability using multiple algorithmic approaches and a production-ready automated data pipeline.

## 📋 Project Overview

This project addresses the automotive retail classification challenge: predicting vehicle acceptability based on financial and physical characteristics. Unlike baseline heuristic evaluations, this study transitions to robust, production-ready machine learning pipelines with rigorous cross-validation and stress-testing protocols.

**Key Challenges:**
- Severe class imbalance (~70% "unacceptable" in training data)
- Categorical feature preservation (ordinal relationships)
- Production-scale generalization (100,000+ record simulation)
- Automated data handling with minimal manual preprocessing

**Highlights:**
- **Champion Algorithm**: XGBoost achieving 98% F1-Macro Score
- **Validation Strategy**: 90-fold Repeated Stratified K-Fold (mitigates sampling luck)
- **Robust Preprocessing**: Modular Scikit-Learn pipeline handling missing data and categorical encoding
- **Stress Testing**: Large-scale evaluation on 100,000 synthetic records

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or later
- `uv` package manager (platform-agnostic)

### Installation & Setup

**1. Install `uv` (if not already installed)**

```bash
# Via pip (recommended)
pip install uv

# Or download from https://github.com/astral-sh/uv
```

**2. Clone and Navigate to Repository**

```bash
git clone <repository-url>
cd a2-classification
```

**3. Install Dependencies**

```bash
uv sync
```

This command reads `pyproject.toml` and installs all required packages in an isolated environment.

**4. Run the Analysis**

```bash
# Execute the main classification pipeline
uv run jupyter lab code/classification.ipynb

# Or run alternative analysis notebooks
uv run jupyter lab code/eda.ipynb              # Exploratory Data Analysis
uv run jupyter lab code/knn_model.ipynb        # Distance-based baseline
uv run jupyter lab code/pipeline.ipynb         # Modular preprocessing demo
```

---

## 📁 Project Structure

```
.
├── data/
│   ├── car.csv                    # Original dataset (1,728 records, 6 features)
│   ├── car.parquet                # Parquet format (efficient processing)
│   └── unknown_car_data.csv       # Test set for stress testing (100,000 records)
│
├── code/
│   ├── car_transformers.py        # Custom Scikit-Learn transformers (pipeline stages)
│   ├── classification.ipynb       # Main comparative analysis (all algorithms)
│   ├── eda.ipynb                  # Exploratory data analysis & feature insights
│   ├── knn_model.ipynb            # K-Nearest Neighbors baseline implementation
│   ├── pipeline.ipynb             # Preprocessing pipeline demonstration
│   └── additional_tests.ipynb     # Supplementary stress tests & diagnostics
│
├── report/
│   ├── main.pdf                   # Final assignment report
│   ├── main.tex                   # LaTeX source
│   ├── chapters/                  # Detailed report sections
│   ├── appendices/                # Supplementary visualizations & results
│   └── figures/                   # Generated plots and logos
│
├── pyproject.toml                 # Project metadata & dependencies
├── README.md                       # This file
└── LICENSE                         # Project license
```

---

## 🔧 Technical Implementation

### Dataset Description

| Feature | Type | Values | Interpretation |
|---------|------|--------|-----------------|
| **Price** | Categorical | low, med, high, vhigh | Buying price |
| **Maintenance** | Categorical | low, med, high, vhigh | Annual maintenance cost |
| **Doors** | Numeric | 2, 3, 4, 5+ | Number of doors |
| **Seats** | Numeric | 2, 4, more (6) | Seating capacity |
| **Storage** | Categorical | small, med, big | Boot size |
| **Safety** | Categorical | low, med, high | Safety rating |
| **Target (shouldBuy)** | Categorical | unacc, acc, good, vgood | **Acceptability** |

**Dataset Statistics:**
- Training: 1,728 instances (90% vault set)
- Vault: 173 instances (holdout validation)
- Stress Test: 100,000 synthetic records

**Class Distribution (Training):**
- Unacceptable (unacc): ~70.02%
- Acceptable (acc): ~22.22%
- Good (good): ~3.99%
- Very Good (vgood): ~3.76%

### Custom Preprocessing Pipeline

The project implements three specialized Scikit-Learn transformers for reproducible data handling:

#### 1. **CarDataCleaner** (`code/car_transformers.py:10-34`)
Standardizes null flavors and string mappings:
```python
# Handles anomalies: "", "null", "NaN", "NULL", "none", "None"
# Maps: "5more" → "6", "more" → "6"
```

#### 2. **CarDataImputer** (`code/car_transformers.py:38-71`)
Implements hybrid imputation strategy:
- **Numeric Columns** (doors, seats): Median imputation
- **Categorical Columns** (price, maintenance, safety, storage): Mode imputation
- Uses `ColumnTransformer` for isolated column handling

#### 3. **CarDataEncoder** (`code/car_transformers.py:75-110`)
Preserves ordinal relationships using custom mappings:
```python
"price": ["low", "med", "high", "vhigh"]
"maintenance": ["low", "med", "high", "vhigh"]
"storage": ["small", "med", "big"]
"safety": ["low", "med", "high"]
```
- Ordinal Encoding maintains feature hierarchy
- Handles unknown categories gracefully

### Modeling Approaches

**Algorithm Families & Performance:**

| Category | Algorithm | F1-Macro (Vault) | Accuracy | Status |
|----------|-----------|------------------|----------|--------|
| **Boosting** | XGBoost | 0.98XX | 0.98XX | ⭐ Champion |
| **Neural** | ANN (MLP) | 0.97XX | 0.97XX | Elite |
| **Bagging** | Random Forest | 0.96XX | 0.96XX | Strong |
| **Kernel** | SVM (RBF) | 0.94XX | 0.94XX | Moderate |
| **Distance** | KNN (K=3) | 0.91XX | 0.91XX | Baseline |
| **Probabilistic** | Naive Bayes | 0.72XX | 0.75XX | Weak |
| **Linear** | SVM (Linear) | 0.6XXX | 0.6XXX | Weak |

**Key Insights:**
- **Tree-based dominance**: XGBoost and Random Forest capture the "hard rules" (low safety → unacc)
- **Feature hierarchy**: safety and seats are dominant decision criteria (information gain: 0.181732 and 0.152259)
- **Stress test reality check**: All models dropped to F1 ≈ 0.25 on 100,000 synthetic records (random baseline: 0.25)

### Validation Strategy

**Repeated Stratified K-Fold Cross-Validation:**
- **Configuration**: 10 repeats × 9 splits = 90 folds
- **Purpose**: Mitigates "sampling luck" inherent in standard 80/20 splits
- **Stratification**: Preserves class distribution across folds
- **Metrics**: F1-Macro (handles imbalance), Accuracy, Precision, Recall

**Holdout "Vault" Set:**
- 10% of training data (173 records) fully sequestered
- Used only for final model evaluation
- Ensures unbiased generalization benchmarking

### Production Stress Testing

**100,000 Record Simulation:**
- Synthetic dataset with same feature distributions but high-volume noise
- Root cause analysis: Training set (1,728 records) insufficient for 60× population scaling
- **Result**: All models converge to random baseline (~0.25 F1)
- **Implication**: Real production requires significantly larger, diverse training corpus

---

## 📊 Results & Key Findings

### Performance Leaderboard (10% Vault Set)

**XGBoost Champion:**
- Perfectly captured override effects (low safety → unacc regardless of price)
- F1-Macro: 0.98XX | Accuracy: 0.98XX
- Execution time: 12.45 ms

**Tree-based methods** showed highest stability across 90-fold CV, while **Naive Bayes** failed due to feature codependency (e.g., Safety=Low always overrides Price).

### Featured Visualizations

- **Figure 1**: Random Forest feature importance (shows safety dominance)
- **Figure 2**: 90-fold model performance distribution (box plot per algorithm)
- **Figure 3**: XGBoost stress test diagnostics (confusion matrix & multi-class ROC)

---

## 🔄 Workflow & Reproducibility

### Data Pipeline Flow

```
Raw Data (car.csv)
    ↓
CarDataCleaner (string standardization)
    ↓
CarDataImputer (missing value handling)
    ↓
CarDataEncoder (ordinal encoding)
    ↓
Train/Validate/Test Split
    ↓
Algorithm Training & 90-fold CV
    ↓
Vault Set Evaluation (final metrics)
    ↓
Stress Test on 100,000 synthetic records
```

### Running Individual Components

```bash
# Preprocessing demonstration
uv run jupyter lab code/pipeline.ipynb

# Full comparative analysis
uv run jupyter lab code/classification.ipynb

# Exploratory insights
uv run jupyter lab code/eda.ipynb

# KNN baseline implementation
uv run jupyter lab code/knn_model.ipynb

# Extended diagnostics
uv run jupyter lab code/additional_tests.ipynb
```

---

## 📦 Dependencies

Managed via `pyproject.toml`:

| Package | Version | Purpose |
|---------|---------|---------|
| **scikit-learn** | ≥1.7.2 | ML algorithms, preprocessing pipelines |
| **xgboost** | ≥3.1.3 | Gradient boosting champion algorithm |
| **pandas** | ≥2.3.3 | Data manipulation & feature engineering |
| **numpy** | ≥2.2.6 | Numerical computing |
| **matplotlib** | ≥3.10.8 | Static visualizations |
| **seaborn** | ≥0.13.2 | Statistical plot styling |
| **pyarrow** | ≥23.0.0 | Parquet format support |
| **jupyterlab** | ≥4.5.2 | Interactive notebook environment |
| **ruff** | ≥0.14.14 | Code formatting & linting |

---

## 🎓 Course Information

| Attribute | Value |
|-----------|-------|
| **Course Number** | MCDA 5580 |
| **Course Title** | Data and Text Mining |
| **Program** | Master of Science in Computing and Data Analytics |
| **Institution** | Saint Mary's University |
| **Department** | Mathematics and Computing Science |
| **Assignment** | Supervised Learning - Classification |
| **Submission Date** | January 26, 2026 |

### Team Members

- **Bhavik Kantilal Bhagat** (A00494758)
- **Jeevan Dhakal** (A00494615)
- **Binziya Siddik** (A00494129)

---

## 📚 Project Outputs

- **Report**: `report/main.pdf` — Comprehensive technical documentation
- **Source Code**: `code/*.ipynb` — Reproducible Jupyter notebooks
- **Data**: `data/` — Training, validation, and stress-test datasets
- **Preprocessing Module**: `code/car_transformers.py` — Reusable pipeline stages

---

## 🤝 Usage & Attribution

This project is submitted as coursework for MCDA 5580 at Saint Mary's University.

**To reuse the preprocessing pipeline:**

```python
from code.car_transformers import CarDataCleaner, CarDataImputer, CarDataEncoder
from sklearn.pipeline import Pipeline

# Build custom preprocessing pipeline
pipeline = Pipeline([
    ('cleaner', CarDataCleaner(feature_names=['price', 'maintenance', 'doors', 'seats', 'storage', 'safety'])),
    ('imputer', CarDataImputer(feature_names=['price', 'maintenance', 'doors', 'seats', 'storage', 'safety'])),
    ('encoder', CarDataEncoder(feature_names=['price', 'maintenance', 'doors', 'seats', 'storage', 'safety'])),
])

X_processed = pipeline.fit_transform(X_raw)
```

---

## 📝 License

[Specify license type if applicable]

---

## ❓ Questions & Contact

For technical questions, refer to:
- **Main Report**: `report/main.pdf`
- **Notebooks**: See inline documentation in `code/*.ipynb`
- **Data Dictionary**: Table 1 in this README

---

**Last Updated**: January 2026
**Status**: Complete & Production-Tested
